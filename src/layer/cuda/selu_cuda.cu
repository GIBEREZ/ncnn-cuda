// SELU CUDA kernel: f(x) = lambda * (x if x > 0 else alpha * (exp(x) - 1))

#include "selu_cuda.h"

namespace ncnn {

__global__ void selu_kernel_cuda(const float* input, float* output, int number,
                                  float alpha, float lambda)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int idxElement = idx * 4;

    if (idxElement + 3 < number) {
        float4 vec_in = *(const float4*)&input[idxElement];
        float4 vec_out;
        vec_out.x = vec_in.x > 0.f ? lambda * vec_in.x : lambda * alpha * (expf(vec_in.x) - 1.f);
        vec_out.y = vec_in.y > 0.f ? lambda * vec_in.y : lambda * alpha * (expf(vec_in.y) - 1.f);
        vec_out.z = vec_in.z > 0.f ? lambda * vec_in.z : lambda * alpha * (expf(vec_in.z) - 1.f);
        vec_out.w = vec_in.w > 0.f ? lambda * vec_in.w : lambda * alpha * (expf(vec_in.w) - 1.f);
        *(float4*)&output[idxElement] = vec_out;
    } else {
        for (int i = 0; i < 4; i++) {
            unsigned int elem_idx = idxElement + i;
            if (elem_idx < number) {
                float val = input[elem_idx];
                output[elem_idx] = val > 0.f ? lambda * val : lambda * alpha * (expf(val) - 1.f);
            }
        }
    }
}

int selu_cuda(const CudaMat& input_blob, CudaMat& output_blob, float alpha, float lambda)
{
    int number = input_blob.total();
    if (output_blob.empty())
        output_blob.create_like(input_blob);

    int threadsPerBlock = 256;
    int totalThreads = (number + 3) / 4;
    int blocksPerGrid = (totalThreads + threadsPerBlock - 1) / threadsPerBlock;

    selu_kernel_cuda<<<blocksPerGrid, threadsPerBlock>>>(
        static_cast<const float*>(input_blob.gpu_data),
        static_cast<float*>(output_blob.gpu_data),
        number, alpha, lambda);

    cudaDeviceSynchronize();
    return 0;
}

int selu_cuda_inplace(CudaMat& input_blob, float alpha, float lambda)
{
    int number = input_blob.total();
    int threadsPerBlock = 256;
    int totalThreads = (number + 3) / 4;
    int blocksPerGrid = (totalThreads + threadsPerBlock - 1) / threadsPerBlock;

    float* gpu_data = static_cast<float*>(input_blob.gpu_data);
    selu_kernel_cuda<<<blocksPerGrid, threadsPerBlock>>>(
        gpu_data, gpu_data, number, alpha, lambda);

    cudaDeviceSynchronize();
    return 0;
}

} // namespace ncnn
