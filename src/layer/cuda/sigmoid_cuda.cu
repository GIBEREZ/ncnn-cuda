// Sigmoid CUDA kernel: f(x) = 1 / (1 + exp(-x))
// Created by GIBEREZ on 2025/12/26.

#include "sigmoid_cuda.h"

namespace ncnn {

__global__ void sigmoid_kernel_cuda(const float* input, float* output, int number)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int idxElement = idx * 4;
    if (idxElement + 3 < number) {
        float4 vec_in = *(const float4*)&input[idxElement];
        float4 vec_out;
        vec_out.x = 1.0f / (1.0f + expf(-vec_in.x));
        vec_out.y = 1.0f / (1.0f + expf(-vec_in.y));
        vec_out.z = 1.0f / (1.0f + expf(-vec_in.z));
        vec_out.w = 1.0f / (1.0f + expf(-vec_in.w));
        *(float4*)&output[idxElement] = vec_out;
    } else {
        for (int i = 0; i < 4; i++) {
            unsigned int elem_idx = idxElement + i;
            if (elem_idx < number) {
                float val = input[elem_idx];
                output[elem_idx] = 1.0f / (1.0f + expf(-val));
            }
        }
    }
}

int sigmoid_cuda(const CudaMat& input_blob, CudaMat& output_blob)
{
    int number = input_blob.total();
    if (output_blob.empty())
    {
        output_blob.create_like(input_blob);
    }

    int totalThreadsNeeded = (number + 3) / 4;
    int threadsPerBlock = 256;
    int blocksPerGrid = (totalThreadsNeeded + threadsPerBlock - 1) / threadsPerBlock;

    sigmoid_kernel_cuda<<<blocksPerGrid, threadsPerBlock>>>(
        static_cast<const float*>(input_blob.gpu_data),
        static_cast<float*>(output_blob.gpu_data),
        number);

    cudaDeviceSynchronize();
    return 0;
}

int sigmoid_cuda_inplace(CudaMat& input_blob)
{
    int number = input_blob.total();
    int totalThreadsNeeded = (number + 3) / 4;
    int threadsPerBlock = 256;
    int blocksPerGrid = (totalThreadsNeeded + threadsPerBlock - 1) / threadsPerBlock;

    float* gpu_data = static_cast<float*>(input_blob.gpu_data);
    sigmoid_kernel_cuda<<<blocksPerGrid, threadsPerBlock>>>(gpu_data, gpu_data, number);

    cudaDeviceSynchronize();
    return 0;
}

} // namespace ncnn