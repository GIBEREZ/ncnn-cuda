//
// Created by GIBEREZ on 2025/12/31.
//
#include "batchnorm_cuda.h"

namespace ncnn {
    __global__ void batchnorm_precompute_kernel(float* a, float* b, const float* bias, const float* slope, const float* mean, const float* var, const float eps, const int Number)
    {
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= Number)
            return;
        unsigned int idxElement = idx * 4;
        float4 var4   = *reinterpret_cast<const float4*>(var + idxElement);
        float4 slope4 = *reinterpret_cast<const float4*>(slope + idxElement);
        float4 mean4  = *reinterpret_cast<const float4*>(mean + idxElement);
        float4 bias4  = *reinterpret_cast<const float4*>(bias + idxElement);

        float4 inv_std4 = make_float4(
            rsqrtf(var4.x + eps),
            rsqrtf(var4.y + eps),
            rsqrtf(var4.z + eps),
            rsqrtf(var4.w + eps)
        );

        float4 b4 = make_float4(
            slope4.x * inv_std4.x,
            slope4.y * inv_std4.y,
            slope4.z * inv_std4.z,
            slope4.w * inv_std4.w
        );

        float4 a4 = make_float4(
            bias4.x - slope4.x * mean4.x * inv_std4.x,
            bias4.y - slope4.y * mean4.y * inv_std4.y,
            bias4.z - slope4.z * mean4.z * inv_std4.z,
            bias4.w - slope4.w * mean4.w * inv_std4.w
        );

        *reinterpret_cast<float4*>(a + idxElement) = a4;
        *reinterpret_cast<float4*>(b + idxElement) = b4;
    }

    __global__ void batchnorm_kernel(float* input, float* a, float* b, const int Number)
    {
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= Number)
            return;
        unsigned int idxElement = idx * 4;

        float4 vec_in = *reinterpret_cast<const float4*>(input + idxElement);
        float4 vec_a  = *reinterpret_cast<const float4*>(a + idxElement);
        float4 vec_b  = *reinterpret_cast<const float4*>(b + idxElement);

        float4 out = make_float4(
             vec_b.x * vec_in.x + vec_a.x,
             vec_b.y * vec_in.y + vec_a.y,
             vec_b.z * vec_in.z + vec_a.z,
             vec_b.w * vec_in.w + vec_a.w
        );

        *reinterpret_cast<float4*>(input + idxElement) = out;
    }

    int BatchNorm_cuda::batchnorm_precompute()
    {
        int threadsPerBlock = 256;
        int vec_size = 16 / 4;
        int totalThreadsNeeded = (channels + vec_size - 1) / vec_size;
        int blocksPerGrid = (totalThreadsNeeded + threadsPerBlock - 1) / threadsPerBlock;
        batchnorm_precompute_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            static_cast<float*>(a_data.gpu_data),
            static_cast<float*>(b_data.gpu_data),
            static_cast<float*>(bias_data.gpu_data),
            static_cast<float*>(slope_data.gpu_data),
            static_cast<float*>(mean_data.gpu_data),
            static_cast<float*>(var_data.gpu_data),
            eps,
            channels
            );
        cudaDeviceSynchronize();
        return 0;
    }

    int BatchNorm_cuda::batchnorm_cuda_inplace(CudaMat& input_blob) const
    {
        int dims = input_blob.dims;
        if (dims == 1)
        {
            int w = input_blob.w;

            int threadsPerBlock = 256;
            int vec_size = 16 / input_blob.elemsize;
            int totalThreadsNeeded = (w + vec_size - 1) / vec_size;
            int blocksPerGrid = (totalThreadsNeeded + threadsPerBlock - 1) / threadsPerBlock;
            batchnorm_kernel<<<blocksPerGrid, threadsPerBlock>>>(static_cast<float*>(input_blob.gpu_data), static_cast<float*>(a_data.gpu_data), static_cast<float*>(b_data.gpu_data), w);
            cudaDeviceSynchronize();

            return 0;
        }
        if (dims == 2)
        {
            int w = input_blob.w;
            int h = input_blob.h;

            int threadsPerBlock = 256;
            int vec_size = 16 / input_blob.elemsize;
            int totalThreadsNeeded = (w + vec_size - 1) / vec_size;
            int blocksPerGrid = (totalThreadsNeeded + threadsPerBlock - 1) / threadsPerBlock;
            batchnorm_kernel<<<blocksPerGrid, threadsPerBlock>>>(static_cast<float*>(input_blob.gpu_data), static_cast<float*>(a_data.gpu_data), static_cast<float*>(b_data.gpu_data), w * h);
            cudaDeviceSynchronize();

            return 0;
        }
        if (dims == 3 || dims == 4)
        {
            int w = input_blob.w;
            int h = input_blob.h;
            int d = input_blob.d;
            int c = input_blob.c;
            int size = w * h * d;

            int threadsPerBlock = 256;
            int vec_size = 16 / input_blob.elemsize;
            int totalThreadsNeeded = (w + vec_size - 1) / vec_size;
            int blocksPerGrid = (totalThreadsNeeded + threadsPerBlock - 1) / threadsPerBlock;
            batchnorm_kernel<<<blocksPerGrid, threadsPerBlock>>>(static_cast<float*>(input_blob.gpu_data), static_cast<float*>(a_data.gpu_data), static_cast<float*>(b_data.gpu_data), size);
            cudaDeviceSynchronize();

            return 0;
        }
        return 0 ;
    }
}