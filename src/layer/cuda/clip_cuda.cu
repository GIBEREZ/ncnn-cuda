//
// Created by GIBEREZ on 2026/1/1.
//
#include "clip_cuda.h"

namespace ncnn {
    __global__ void clip_kernel_cuda(const float* input, float* output, int number, float min, float max)
    {
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int idxElement = idx * 4;
        if (idxElement + 3 < number) {
            float4 vec_in = *(const float4*)&input[idxElement];
            float4 vec_out;
            vec_out.x = fminf(fmaxf(vec_in.x, min), max);
            vec_out.y = fminf(fmaxf(vec_in.y, min), max);
            vec_out.z = fminf(fmaxf(vec_in.z, min), max);
            vec_out.w = fminf(fmaxf(vec_in.w, min), max);
            *(float4*)&output[idxElement] = vec_out;
        } else {
            for (int i = 0; i < 4; i++) {
                int pos = idxElement + i;
                if (pos < number) {
                    float val = input[pos];
                    output[pos] = fminf(fmaxf(val, min), max);
                }
            }
        }
    }

    __global__ void clip_inplace_kernel(float* data, int number, float min, float max)
    {
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int idxElement = idx * 4;
        if (idxElement + 3 < number) {
            float4 vec_in = *(float4*)data;
            // Manual indexed access
            float4 vec_out;
            float* ptr = data + idxElement;
            vec_out.x = fminf(fmaxf(ptr[0], min), max);
            vec_out.y = fminf(fmaxf(ptr[1], min), max);
            vec_out.z = fminf(fmaxf(ptr[2], min), max);
            vec_out.w = fminf(fmaxf(ptr[3], min), max);
            *(float4*)(data + idxElement) = vec_out;
        } else {
            for (int i = 0; i < 4; i++) {
                int pos = idxElement + i;
                if (pos < number) {
                    data[pos] = fminf(fmaxf(data[pos], min), max);
                }
            }
        }
    }

    int clip_cuda(const CudaMat& input_blob, CudaMat& output_blob, float min, float max)
    {
        int number = input_blob.total();
        if (output_blob.empty())
            output_blob.create_like(input_blob);

        int threadsPerBlock = 256;
        int totalThreads = (number + 3) / 4;
        int blocksPerGrid = (totalThreads + threadsPerBlock - 1) / threadsPerBlock;
        clip_kernel_cuda<<<blocksPerGrid, threadsPerBlock>>>(
            static_cast<const float*>(input_blob.gpu_data),
            static_cast<float*>(output_blob.gpu_data),
            number, min, max);
        cudaDeviceSynchronize();
        return 0;
    }

    int clip_cuda_inplace(CudaMat& input_blob, float min, float max)
    {
        int number = input_blob.total();
        int threadsPerBlock = 256;
        int totalThreads = (number + 3) / 4;
        int blocksPerGrid = (totalThreads + threadsPerBlock - 1) / threadsPerBlock;
        clip_inplace_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            static_cast<float*>(input_blob.gpu_data), number, min, max);
        cudaDeviceSynchronize();
        return 0;
    }
}