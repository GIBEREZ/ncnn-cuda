//
// Created by GIBEREZ on 2025/12/31.
//
#include "absval_cuda.h"

namespace ncnn {
    __global__ void AbsVal_kernel(const float* input, float* output, int Number)
    {
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= Number)
            return;
        unsigned int idxElement = idx * 4;
        const float4 vec_in = *reinterpret_cast<const float4*>(input + idxElement);
        float4 vec_out;
        vec_out.x = fabsf(vec_in.x);
        vec_out.y = fabsf(vec_in.y);
        vec_out.z = fabsf(vec_in.z);
        vec_out.w = fabsf(vec_in.w);
        *reinterpret_cast<float4*>(output + idxElement) = vec_out;
    }

    __global__ void AbsVal_inplace_kernel(float* input, int Number)
    {
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= Number)
            return;
        unsigned int idxElement = idx * 4;
        const float4 vec_in = *reinterpret_cast<const float4*>(input + idxElement);
        float4 vec_out;
        vec_out.x = fabsf(vec_in.x);
        vec_out.y = fabsf(vec_in.y);
        vec_out.z = fabsf(vec_in.z);
        vec_out.w = fabsf(vec_in.w);
        *reinterpret_cast<float4*>(input + idxElement) = vec_out;
    }

    void absval_cuda_inplace(CudaMat& input_blob)
    {
        int w = input_blob.w;
        int h = input_blob.h;
        int channels = input_blob.c;
        int size = w * h;

        int threadsPerBlock = 256;
        int vec_size = 16 / input_blob.elemsize;
        int totalThreadsNeeded = (size + vec_size - 1) / vec_size;
        int blocksPerGrid = (totalThreadsNeeded + threadsPerBlock - 1) / threadsPerBlock;

        AbsVal_inplace_kernel<<<blocksPerGrid, threadsPerBlock>>>(static_cast<float*>(input_blob.gpu_data), size);
        cudaDeviceSynchronize();
    }
}