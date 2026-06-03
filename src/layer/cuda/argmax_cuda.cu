//
// Created by GIBEREZ on 2025/12/31.
//
#include <cub/cub.cuh>
#include "argmax_cuda.h"

namespace ncnn {

/**
 * @brief Fill array with sequential values [0, 1, 2, ..., size-1] as float.
 */
__global__ void fill_sequence_kernel(float* data, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        data[idx] = static_cast<float>(idx);
}

/**
 * @brief Convert float array to unsigned int keys suitable for radix sort.
 *        Uses the standard sign-bit-flip trick so that unsigned ordering
 *        matches float numeric ordering (handles negative values correctly).
 */
__global__ void float_to_sortable_uint_kernel(const float* input, unsigned int* output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        unsigned int u = __float_as_uint(input[idx]);
        // Flip sign bit for positive, flip all bits for negative
        output[idx] = (u & 0x80000000) ? ~u : (u ^ 0x80000000);
    }
}

/**
 * @brief Gather top-k values from the original input using sorted indices.
 *        out_values[i] = input[static_cast<int>(indices[i])]
 */
__global__ void gather_topk_values_kernel(const float* input, const float* indices, float* out_values, int k)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < k)
    {
        int src_idx = static_cast<int>(indices[idx]);
        out_values[idx] = input[src_idx];
    }
}

/**
 * @brief Copy a single value from input[index[0]] to output[0].
 *        Used for the topk==1 && out_max_val case.
 */
__global__ void copy_value_at_index_kernel(const float* input, const float* index_ptr, float* output)
{
    int src_idx = static_cast<int>(index_ptr[0]);
    output[0] = input[src_idx];
}

int ArgMax_cuda::argmax_cuda(const CudaMat& input_blob, CudaMat& output_blob) const
{
    int size = input_blob.total();

    // Clamp topk to actual data size
    int k = topk;
    if (k > size)
        k = size;
    if (k <= 0)
        return -100;

    // Allocate output blob
    // Layout: w = k, h = 1 (indices only) or h = 2 (values + indices)
    if (out_max_val)
        output_blob.create(k, 2, 4);
    else
        output_blob.create(k, 1, 4);
    if (output_blob.empty())
        return -100;

    const float* d_input = static_cast<const float*>(input_blob.gpu_data);
    float* d_output = static_cast<float*>(output_blob.gpu_data);

    // -----------------------------------------------------------------
    // Case 1: topk == 1  —  use CUB DeviceReduce::ArgMax (fast path)
    // -----------------------------------------------------------------
    if (k == 1)
    {
        void* d_temp_storage = nullptr;
        size_t temp_storage_bytes = 0;

        // d_output layout when out_max_val: [value, index] i.e. w=1, h=2
        // d_output layout when !out_max_val: [index]          i.e. w=1, h=1
        float* d_index_ptr = out_max_val ? (d_output + 1) : d_output;

        // Determine temp storage size
        cub::DeviceReduce::ArgMax(d_temp_storage, temp_storage_bytes, d_input, d_index_ptr, size);

        // Allocate temp storage
        cudaMalloc(&d_temp_storage, temp_storage_bytes);

        // Perform argmax reduction
        cub::DeviceReduce::ArgMax(d_temp_storage, temp_storage_bytes, d_input, d_index_ptr, size);

        cudaFree(d_temp_storage);

        // If out_max_val, also write the max value to d_output[0]
        if (out_max_val)
        {
            copy_value_at_index_kernel<<<1, 1>>>(d_input, d_output + 1, d_output);
        }

        cudaDeviceSynchronize();
        return 0;
    }

    // -----------------------------------------------------------------
    // Case 2: topk > 1  —  use CUB DeviceRadixSort on key-value pairs
    // -----------------------------------------------------------------

    // Step 1: Allocate temporary GPU arrays
    unsigned int* d_keys_uint;   // sortable unsigned int keys (transformed float)
    float*        d_indices;     // original indices [0, 1, ..., size-1]
    cudaMalloc(&d_keys_uint, size * sizeof(unsigned int));
    cudaMalloc(&d_indices, size * sizeof(float));

    // Step 2: Fill d_indices with [0, 1, 2, ..., size-1]
    {
        int threadsPerBlock = 256;
        int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
        fill_sequence_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_indices, size);
        cudaDeviceSynchronize();
    }

    // Step 3: Convert float input to sortable unsigned int keys
    {
        int threadsPerBlock = 256;
        int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
        float_to_sortable_uint_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_keys_uint, size);
        cudaDeviceSynchronize();
    }

    // Step 4: Sort (key, value) pairs descending by key
    {
        void* d_temp_storage = nullptr;
        size_t temp_storage_bytes = 0;

        cub::DeviceRadixSort::SortPairsDescending(
            d_temp_storage, temp_storage_bytes,
            d_keys_uint, d_indices, size);

        cudaMalloc(&d_temp_storage, temp_storage_bytes);

        cub::DeviceRadixSort::SortPairsDescending(
            d_temp_storage, temp_storage_bytes,
            d_keys_uint, d_indices, size);

        cudaDeviceSynchronize();
        cudaFree(d_temp_storage);
    }

    // Step 5: Extract top-k results to output
    if (out_max_val)
    {
        // d_output[0 .. k-1]   = top-k values (gathered from input)
        // d_output[k .. 2*k-1] = top-k indices
        int threadsPerBlock = 256;
        int blocksPerGrid = (k + threadsPerBlock - 1) / threadsPerBlock;
        gather_topk_values_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_input, d_indices, d_output, k);
        cudaDeviceSynchronize();

        // Copy top-k indices to d_output + k
        cudaMemcpy(d_output + k, d_indices, k * sizeof(float), cudaMemcpyDeviceToDevice);
    }
    else
    {
        // d_output[0 .. k-1] = top-k indices
        cudaMemcpy(d_output, d_indices, k * sizeof(float), cudaMemcpyDeviceToDevice);
    }

    // Step 6: Cleanup
    cudaFree(d_keys_uint);
    cudaFree(d_indices);

    cudaDeviceSynchronize();
    return 0;
}

} // namespace ncnn
