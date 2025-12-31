//
// Created by GIBEREZ on 2025/12/31.
//
#include <cub/cub.cuh>
#include "argmax_cuda.h"

namespace ncnn {
    __global__ void block_topk_kernel(const float* input, float* output, int K, int Number)
    {
        __shared__ float out[32];
        unsigned int idx = (blockDim.x * blockIdx.x + threadIdx.x) * 4;  // 这里要乘以4
        unsigned int warpId = threadIdx.x / warpSize;   // 当前线程位于第几个warp
        unsigned int laneId = threadIdx.x % warpSize;   // 当前线程是warp中的第几个线程

        if (idx < Number)
            return;

        const float4 vec_in = *reinterpret_cast<const float4*>(input);

        #pragma unroll
        for (int offset = warpSize >> 1; offset > 0; offset >>= 1)
        {

        }
    }

    __global__ void block_topk_outmax_kernel(const float* input, float* output, int topk, int Number)
    {
        __shared__ float out[topk];
        __shared__ float val[topk];
    }

    int ArgMax_cuda::argmax_cuda(const CudaMat& input_blob, CudaMat& output_blob) const
    {
        int size = input_blob.total();

        if (out_max_val)
            output_blob.create(topk, 2, 4);
        else
            output_blob.create(topk, 1, 4);
        if (output_blob.empty())
            return -100;

        if (topk == 1)
        {
            void* d_temp_storage = nullptr;
            size_t temp_storage_bytes = 0;
            cub::DeviceReduce::ArgMax(
                d_temp_storage, temp_storage_bytes,
                static_cast<float*>(input_blob.gpu_data),
                out_max_val ? output_blob.gpu_data() + 1 : nullptr,
                static_cast<int*>(output_blob.gpu_data),
                size
                );
            cudaMalloc(&d_temp_storage, temp_storage_bytes);
            cub::DeviceReduce::ArgMax(
                d_temp_storage, temp_storage_bytes,
                static_cast<float*>(input_blob.gpu_data),
                out_max_val ? output_blob.gpu_data() + 1 : nullptr,
                static_cast<int*>(output_blob.gpu_data),
                size
                );
        }
        else
        {

        }

        return 0;
    }
}
