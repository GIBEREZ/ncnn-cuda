//
// Created by GIBEREZ on 2026/1/1.
//
#include "bnll_cuda.h"

namespace ncnn {
    __device__ __forceinline__ float bnll_activation(float x) {
        if (x > 0.0f) {
            if (x > 20.0f) return x;
            return x + log1pf(expf(-x));
        }
        if (x < -20.0f) return 0.0f;
        return log1pf(expf(x));
    }

    __device__ __forceinline__ float bnll_activation_fast(float x) {
        const float threshold = 20.0f;
        if (x > threshold) return x;
        if (x < -threshold) return 0.0f;

        return x > 0.0f ? x + log1pf(__expf(-x)) : log1pf(__expf(x));
    }

    __global__ void BNLL_inplace_kernel(float* input, const int Number)
    {
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int vec_idx = idx * 4;

        if (vec_idx + 3 < Number) {
            const float4 vec_in = *reinterpret_cast<const float4*>(input + vec_idx);

            float4 vec_out;
            vec_out.x = bnll_activation(vec_in.x);
            vec_out.y = bnll_activation(vec_in.y);
            vec_out.z = bnll_activation(vec_in.z);
            vec_out.w = bnll_activation(vec_in.w);

            *reinterpret_cast<float4*>(input + vec_idx) = vec_out;
        }
        else {
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int pos = vec_idx + i;
                if (pos < Number) {
                    input[pos] = bnll_activation(input[pos]);
                }
            }
        }
    }

    int bnll_cuda_inplace(CudaMat& input_blob)
    {
        int size = input_blob.total();

        int threadsPerBlock = 256;
        int vec_size = 16 / input_blob.elemsize;
        int totalThreadsNeeded = (size + vec_size - 1) / vec_size;
        int blocksPerGrid = (totalThreadsNeeded + threadsPerBlock - 1) / threadsPerBlock;

        BNLL_inplace_kernel<<<blocksPerGrid, threadsPerBlock>>>(static_cast<float*>(input_blob.gpu_data), size);
        cudaDeviceSynchronize();

        return 0;
    }
}