//
// Created by GIBEREZ on 2025/12/31.
//
#include <cub/cub.cuh>
#include "argmax_cuda.h"

namespace ncnn {
    __global__ void block_topk_kernel(const float* input, float* output, int K, int Number)
    {
        __shared__ float out[32];
        unsigned int idx = (blockDim.x * blockIdx.x + threadIdx.x) * 4;  // multiply by 4
        unsigned int warpId = threadIdx.x / warpSize;   // warp index
        unsigned int laneId = threadIdx.x % warpSize;   // lane index within warp

        if (idx < Number)
            return;

        const float4 vec_in = *reinterpret_cast<const float4*>(input);

        #pragma unroll
        for (int offset = warpSize >> 1; offset > 0; offset >>= 1)
        {

        }
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

        }
        else
        {

        }

        return 0;
    }
}
