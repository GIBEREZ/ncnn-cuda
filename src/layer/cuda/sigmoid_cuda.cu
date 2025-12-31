//
// Created by GIBEREZ on 2025/12/26.
//
#include "sigmoid_cuda.h"

namespace ncnn {
    __global__ void Sigmoid_kernel(float* x, int N) {
        if (const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N)
        {
            x[idx] = 1.0f / (1.0f + expf(-x[idx]));
        }
    }

    __global__ void Sigmoid_float4_kernel(float* x, int N) {
        const unsigned int idx = (blockDim.x * blockIdx.x + threadIdx.x) * 4;
        const unsigned int idxElement = idx * 4;
        if (idx < N) {
            float4 tmp_x = *reinterpret_cast<float4*>(&x[idxElement]);
            float4 vec_out;
            vec_out.x = 1.0f / (1.0f + expf(-tmp_x.x));
            vec_out.y = 1.0f / (1.0f + expf(-tmp_x.y));
            vec_out.z = 1.0f / (1.0f + expf(-tmp_x.z));
            vec_out.w = 1.0f / (1.0f + expf(-tmp_x.w));
            *reinterpret_cast<float4*>(&x[idxElement]) = vec_out;
        }
    }

    int sigmoid_cuda_inplace(CudaMat& input_blob)
    {
        int total = input_blob.total();
        int total4 = (total + 3) / 4;

        int threads_per_block = 256;
        int blocks_per_grid = (total4 + threads_per_block - 1) / threads_per_block;

        Sigmoid_float4_kernel<<<blocks_per_grid, threads_per_block>>>(static_cast<float*>(input_blob.gpu_data), total);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "CUDA error in Sigmoid_kernel: %s\n", cudaGetErrorString(err));
            return -1;
        }

        cudaDeviceSynchronize();
        return 0;
    }
} // namespace ncnn