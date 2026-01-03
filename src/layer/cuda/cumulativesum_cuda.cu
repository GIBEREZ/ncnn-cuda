//
// Created by GIBEREZ on 2026/1/1.
//
#include "cumulativesum_cuda.h"
#include <cub/cub.cuh>

namespace ncnn {
    __global__ void ReducePartSum_kernel(const float* input, float* part_output, size_t Number)
    {
        extern __shared__ float sum[];
        int tid = threadIdx.x;

        unsigned int idx = blockIdx.x * blockDim.x + tid;
        unsigned int idxElement = idx * 4;

        float threadSum = 0.0f;
        if (idxElement < Number) {
            size_t remain = Number - idxElement;
            float4 vec_in = make_float4(0,0,0,0);
            if (remain >= 4) {
                vec_in = *reinterpret_cast<const float4*>(input + idxElement);
            } else {
                if (remain > 0) vec_in.x = input[idxElement + 0];
                if (remain > 1) vec_in.y = input[idxElement + 1];
                if (remain > 2) vec_in.z = input[idxElement + 2];
            }
            threadSum = vec_in.x + vec_in.y + vec_in.z + vec_in.w;
        }

        sum[tid] = threadSum;
        __syncthreads();

        size_t stride = blockDim.x / 2;
        while (stride > 0) {
            if (tid < stride) {
                sum[tid] += sum[tid + stride];
            }
            __syncthreads();
            stride /= 2;
        }

        if (tid == 0) part_output[blockIdx.x] = sum[0];
    }

    __global__ void AddBlockPrefix_inplace_kernel(float* input, const float* part_output, size_t Number)
    {

        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int idxElement = idx * 4;
        if (idxElement >= Number)
            return;
        float4 vec_in = *reinterpret_cast<const float4*>(input + idxElement);
        float base = (blockIdx.x == 0) ? 0.0f : part_output[blockIdx.x - 1];
        float4 vec_out;
        vec_out.x = vec_in.x;
        vec_out.y = vec_in.y + vec_out.x;
        vec_out.z = vec_in.z + vec_out.y;
        vec_out.w = vec_in.w + vec_out.z;
        vec_out.x += base;
        vec_out.y += base;
        vec_out.z += base;
        vec_out.w += base;
        *reinterpret_cast<float4*>(input + idxElement) = vec_out;
    }

    int cumulativesum_cuda_inplace(CudaMat& input_blob, int axis)
    {
        int dims = input_blob.dims;
        int positive_axis = axis < 0 ? dims + axis : axis;

        if (dims == 1)
        {
            void* d_temp_storage = nullptr;
            size_t temp_storage_bytes = 0;

            cub::DeviceScan::InclusiveSum(
                d_temp_storage,
                temp_storage_bytes,
                input_blob.gpu_data,
                input_blob.total()
                );

            cudaMalloc(&d_temp_storage, temp_storage_bytes);

            cub::DeviceScan::InclusiveSum(
                d_temp_storage,
                temp_storage_bytes,
                input_blob.gpu_data,
                input_blob.total()
                );

            return 0;
        }
        if (dims == 2 && positive_axis == 0)
        {
            int size = input_blob.w * input_blob.h;

            int threadsPerBlock = 256;
            int vec_size = 16 / input_blob.elemsize;
            int totalThreadsNeeded = (size + vec_size - 1) / vec_size;
            int blocksPerGrid = (totalThreadsNeeded + threadsPerBlock - 1) / threadsPerBlock;
            float* part_output = nullptr;
            cudaMalloc(&part_output, blocksPerGrid * sizeof(float));
            size_t sharedBytes = threadsPerBlock * sizeof(float);
            ReducePartSum_kernel<<<blocksPerGrid, threadsPerBlock, sharedBytes>>>(static_cast<float*>(input_blob.gpu_data), part_output, size);
            cudaDeviceSynchronize();

            void* d_temp_storage = nullptr;
            size_t temp_storage_bytes = 0;
            cub::DeviceScan::InclusiveSum(
                d_temp_storage,
                temp_storage_bytes,
                part_output,
                part_output,
                blocksPerGrid
            );
            cudaMalloc(&d_temp_storage, temp_storage_bytes);
            cub::DeviceScan::InclusiveSum(
                d_temp_storage,
                temp_storage_bytes,
                part_output,
                part_output,
                blocksPerGrid
            );
            cudaFree(d_temp_storage);
            AddBlockPrefix_inplace_kernel<<<blocksPerGrid, threadsPerBlock>>>(static_cast<float*>(input_blob.gpu_data), part_output, size);
            cudaDeviceSynchronize();
        }
        if (dims == 2 && positive_axis == 1)
        {
            int w = input_blob.w;
            int h = input_blob.h;

            void* d_temp_storage = nullptr;
            size_t temp_storage_bytes = 0;

            cub::DeviceScan::InclusiveSum(
                d_temp_storage,
                temp_storage_bytes,
                static_cast<float*>(nullptr),
                static_cast<float*>(nullptr),
                w
            );

            cudaMalloc(&d_temp_storage, temp_storage_bytes);

            #pragma omp parallel for
            for (int i = 0; i < h; ++i)
            {
                float* row_ptr = static_cast<float*>(input_blob.data) + i * w;

                cub::DeviceScan::InclusiveSum(
                    d_temp_storage,
                    temp_storage_bytes,
                    row_ptr,
                    row_ptr,
                    w
                );
            }
            cudaFree(d_temp_storage);
            return 0;
        }
        if (dims == 3 && positive_axis == 0)
        {
            int w = input_blob.w;
            int h = input_blob.h;
            int c = input_blob.c;
            int size = w * h;


        }

        return 0;
    }
}