//
// Created by GIBEREZ on 2026/1/1.
//
#include "cumulativesum_cuda.h"
#include <cub/cub.cuh>

namespace ncnn {

    /**
     * @brief Generic cumulative sum kernel for all dim/axis combinations.
     *
     * For a tensor with axis = k:
     *   - outer_size = product of dimensions before axis k
     *   - ax_size    = dim[axis] (number of elements to cumulatively sum)
     *   - inner_size = product of dimensions after axis k (= stride between consecutive axis elements)
     *
     * Each thread handles one independent cumulative sum segment (sequential scan).
     *
     * @param data        Pointer to GPU data array
     * @param outer_size  Product of dimensions before the target axis
     * @param ax_size     Size of the target axis (segment length)
     * @param inner_size  Product of dimensions after the target axis (segment stride)
     */
    __global__ void cumulativesum_kernel(float* data, int outer_size, int ax_size, int inner_size)
    {
        int num_segments = outer_size * inner_size;
        int seg_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (seg_idx >= num_segments) return;

        // Decompose segment index into outer/inner parts
        int outer_idx = seg_idx / inner_size;
        int inner_idx = seg_idx % inner_size;

        // Base pointer for this segment
        float* base = data + outer_idx * ax_size * inner_size + inner_idx;

        // Sequential inclusive cumulative sum along the axis
        float accum = 0.0f;
        for (int i = 0; i < ax_size; i++)
        {
            accum += base[i * inner_size];
            base[i * inner_size] = accum;
        }
    }

    int cumulativesum_cuda_inplace(CudaMat& input_blob, int axis)
    {
        int dims = input_blob.dims;
        int positive_axis = axis < 0 ? dims + axis : axis;

        float* data = static_cast<float*>(input_blob.gpu_data);
        int threadsPerBlock = 256;

        // ---- dims == 1: single 1D array, use efficient CUB scan ----
        if (dims == 1)
        {
            void* d_temp_storage = nullptr;
            size_t temp_storage_bytes = 0;

            cub::DeviceScan::InclusiveSum(
                d_temp_storage,
                temp_storage_bytes,
                data,
                input_blob.total());

            cudaMalloc(&d_temp_storage, temp_storage_bytes);

            cub::DeviceScan::InclusiveSum(
                d_temp_storage,
                temp_storage_bytes,
                data,
                input_blob.total());

            cudaFree(d_temp_storage);

            return 0;
        }

        // ---- Compute segment parameters from dims and axis ----
        int outer_size = 1;
        int ax_size = 1;
        int inner_size = 1; // stride between consecutive axis elements

        if (dims == 2)
        {
            int w = input_blob.w;
            int h = input_blob.h;

            if (positive_axis == 0)
            {
                // sum over rows: per-column prefix sum (h elements, stride = w)
                outer_size = 1;
                ax_size = h;
                inner_size = w;
            }
            else // positive_axis == 1
            {
                // sum over columns: per-row prefix sum (w elements, contiguous stride = 1)
                outer_size = h;
                ax_size = w;
                inner_size = 1;
            }
        }
        else if (dims == 3)
        {
            int w = input_blob.w;
            int h = input_blob.h;
            int c = input_blob.c;

            if (positive_axis == 0)
            {
                // sum over channels: per-position prefix sum (c elements, stride = h*w)
                outer_size = 1;
                ax_size = c;
                inner_size = h * w;
            }
            else if (positive_axis == 1)
            {
                // sum over rows within each channel (h elements, stride = w)
                outer_size = c;
                ax_size = h;
                inner_size = w;
            }
            else // positive_axis == 2
            {
                // sum over columns within each channel (w elements, contiguous stride = 1)
                outer_size = c * h;
                ax_size = w;
                inner_size = 1;
            }
        }
        else if (dims == 4)
        {
            int w = input_blob.w;
            int h = input_blob.h;
            int d = input_blob.d;
            int c = input_blob.c;

            if (positive_axis == 0)
            {
                // sum over channels: per-position prefix sum (c elements, stride = d*h*w)
                outer_size = 1;
                ax_size = c;
                inner_size = d * h * w;
            }
            else if (positive_axis == 1)
            {
                // sum over depth within each channel (d elements, stride = h*w)
                outer_size = c;
                ax_size = d;
                inner_size = h * w;
            }
            else if (positive_axis == 2)
            {
                // sum over rows within each depth slice (h elements, stride = w)
                outer_size = c * d;
                ax_size = h;
                inner_size = w;
            }
            else // positive_axis == 3
            {
                // sum over columns within each depth slice (w elements, contiguous stride = 1)
                outer_size = c * d * h;
                ax_size = w;
                inner_size = 1;
            }
        }

        int num_segments = outer_size * inner_size;
        int blocksPerGrid = (num_segments + threadsPerBlock - 1) / threadsPerBlock;

        cumulativesum_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            data, outer_size, ax_size, inner_size);
        cudaDeviceSynchronize();

        return 0;
    }

} // namespace ncnn