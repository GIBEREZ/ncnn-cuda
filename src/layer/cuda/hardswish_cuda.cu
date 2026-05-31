// HardSwish CUDA kernel: y = x * clamp(alpha * x + beta, 0, 1)

#include "hardswish_cuda.h"

namespace ncnn {

__global__ void hardswish_kernel_cuda(const float* input, float* output, int number,
                                       float alpha, float beta)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int idxElement = idx * 4;
    if (idxElement + 3 < number) {
        float4 vec_in = *(const float4*)&input[idxElement];
        float4 vec_out;
        float t = vec_in.x * alpha + beta; vec_out.x = vec_in.x * fminf(fmaxf(t, 0.f), 1.f);
        t = vec_in.y * alpha + beta;        vec_out.y = vec_in.y * fminf(fmaxf(t, 0.f), 1.f);
        t = vec_in.z * alpha + beta;        vec_out.z = vec_in.z * fminf(fmaxf(t, 0.f), 1.f);
        t = vec_in.w * alpha + beta;        vec_out.w = vec_in.w * fminf(fmaxf(t, 0.f), 1.f);
        *(float4*)&output[idxElement] = vec_out;
    } else {
        for (int i = 0; i < 4; i++) {
            unsigned int elem_idx = idxElement + i;
            if (elem_idx < number) {
                float x = input[elem_idx];
                float t = x * alpha + beta;
                output[elem_idx] = x * fminf(fmaxf(t, 0.f), 1.f);
            }
        }
    }
}

int hardswish_cuda(const CudaMat& input_blob, CudaMat& output_blob, float alpha, float beta)
{
    int number = input_blob.total();
    if (output_blob.empty())
        output_blob.create_like(input_blob);

    int threadsPerBlock = 256;
    int totalThreads = (number + 3) / 4;
    int blocksPerGrid = (totalThreads + threadsPerBlock - 1) / threadsPerBlock;

    hardswish_kernel_cuda<<<blocksPerGrid, threadsPerBlock>>>(
        static_cast<const float*>(input_blob.gpu_data),
        static_cast<float*>(output_blob.gpu_data),
        number, alpha, beta);

    cudaDeviceSynchronize();
    return 0;
}

int hardswish_cuda_inplace(CudaMat& input_blob, float alpha, float beta)
{
    int number = input_blob.total();
    int threadsPerBlock = 256;
    int totalThreads = (number + 3) / 4;
    int blocksPerGrid = (totalThreads + threadsPerBlock - 1) / threadsPerBlock;

    float* gpu_data = static_cast<float*>(input_blob.gpu_data);
    hardswish_kernel_cuda<<<blocksPerGrid, threadsPerBlock>>>(
        gpu_data, gpu_data, number, alpha, beta);

    cudaDeviceSynchronize();
    return 0;
}

} // namespace ncnn
