//
// Created by GIBEREZ on 2026/1/1.
//
#include "concat_cuda.h"

namespace ncnn {
    __global__ void Concat_dims2_axis1_kernel(const float* input, float* output, int input_blob_index, int input_w, int offset, int top_w, int Number)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx >= Number) return;

        int row = idx / top_w;
        int col = idx % top_w;

        if (col >= offset && col < offset + input_w)
        {
            int local_col = col - offset;
            int input_idx = row * input_w + local_col;
            output[idx] = input[input_idx];
        }
    }

    void Concat_dims2_axis1(const CudaMat& input_blob, CudaMat& output_blob, int h, int input_blob_index, int offset, int top_w)
    {
        int input_w = input_blob.w;
        int threadsPerBlock = 256;
        int blocksPerGrid = (h * input_w + threadsPerBlock - 1) / threadsPerBlock;

        Concat_dims2_axis1_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            static_cast<const float*>(input_blob.gpu_data),
            static_cast<float*>(output_blob.gpu_data),
            input_blob_index,
            input_w,
            offset,
            top_w,
            h * top_w
        );
        cudaDeviceSynchronize();
    }
}