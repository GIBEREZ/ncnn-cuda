//softmax_cuda(input_blob, output_blob);
// Created by GIBEREZ on 2025/12/26.
//

#include "softmax_cuda.h"

namespace ncnn{

    __global__ void softmax_kernel(const float* input, float* output, int Number)
    {

    }

    __global__ void softmax_kernel_inplace(float* input, int Number)
    {

    }

    int softmax_cuda(const CudaMat& input_blob, CudaMat& output_blob)
    {
        return 0;
    }

    int softmax_cuda_inplace(CudaMat& input_blob)
    {

        return 0;
    }
}// namespace ncnn