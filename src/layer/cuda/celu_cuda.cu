//
// Created by GIBEREZ on 2026/1/1.
//
#include "celu_cuda.h"

namespace ncnn {
    __global__ void celu_inplace_kernel(float* input, float alpha, int Number)
    {
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx * 4 >= Number)
            return;
        unsigned int idxElement = idx * 4;

        if (idxElement + 3 < Number) {
            const float4 vec_in = *reinterpret_cast<const float4*>(input + idxElement);

            float4 vec_out = make_float4(
                (expf(vec_in.x / alpha) - 1.f) * alpha,
                (expf(vec_in.y / alpha) - 1.f) * alpha,
                (expf(vec_in.z / alpha) - 1.f) * alpha,
                (expf(vec_in.w / alpha) - 1.f) * alpha
            );

            *reinterpret_cast<float4*>(input + idxElement) = vec_out;
        }
        else {
        #pragma unroll
            for (int i = 0; i < 4; i++) {
                int pos = idxElement + i;
                if (pos < Number) {
                    input[pos] = (expf(input[pos] / alpha) - 1.f) * alpha;
                }
            }
        }
    }

    int celu_cuda_inplace(CudaMat& input_blob, float alpha)
    {
        int threadsPerBlock = 256;
        int vec_size = 16 / input_blob.elemsize;
        int totalThreadsNeeded = (input_blob.total() + vec_size - 1) / vec_size;
        int blocksPerGrid = (totalThreadsNeeded + threadsPerBlock - 1) / threadsPerBlock;
        celu_inplace_kernel<<<blocksPerGrid, threadsPerBlock>>>(static_cast<float*>(input_blob.gpu_data), alpha, input_blob.total());
        cudaDeviceSynchronize();
        return 0;
    }
}