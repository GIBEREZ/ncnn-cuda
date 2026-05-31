// Mish CUDA kernel: y = x * tanh(softplus(x)), softplus(x) = ln(1 + exp(x))

#include "mish_cuda.h"

namespace ncnn {

__global__ void mish_kernel_cuda(const float* input, float* output, int number)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int idxElement = idx * 4;
    if (idxElement + 3 < number) {
        float4 vec_in = *(const float4*)&input[idxElement];
        float4 vec_out;
        vec_out.x = vec_in.x * tanhf(logf(expf(vec_in.x) + 1.f));
        vec_out.y = vec_in.y * tanhf(logf(expf(vec_in.y) + 1.f));
        vec_out.z = vec_in.z * tanhf(logf(expf(vec_in.z) + 1.f));
        vec_out.w = vec_in.w * tanhf(logf(expf(vec_in.w) + 1.f));
        *(float4*)&output[idxElement] = vec_out;
    } else {
        for (int i = 0; i < 4; i++) {
            unsigned int elem_idx = idxElement + i;
            if (elem_idx < number) {
                float x = input[elem_idx];
                output[elem_idx] = x * tanhf(logf(expf(x) + 1.f));
            }
        }
    }
}

int mish_cuda(const CudaMat& input_blob, CudaMat& output_blob)
{
    int number = input_blob.total();
    if (output_blob.empty())
        output_blob.create_like(input_blob);

    int threadsPerBlock = 256;
    int totalThreads = (number + 3) / 4;
    int blocksPerGrid = (totalThreads + threadsPerBlock - 1) / threadsPerBlock;

    mish_kernel_cuda<<<blocksPerGrid, threadsPerBlock>>>(
        static_cast<const float*>(input_blob.gpu_data),
        static_cast<float*>(output_blob.gpu_data),
        number);

    cudaDeviceSynchronize();
    return 0;
}

int mish_cuda_inplace(CudaMat& input_blob)
{
    int number = input_blob.total();
    int threadsPerBlock = 256;
    int totalThreads = (number + 3) / 4;
    int blocksPerGrid = (totalThreads + threadsPerBlock - 1) / threadsPerBlock;

    float* gpu_data = static_cast<float*>(input_blob.gpu_data);
    mish_kernel_cuda<<<blocksPerGrid, threadsPerBlock>>>(gpu_data, gpu_data, number);

    cudaDeviceSynchronize();
    return 0;
}

} // namespace ncnn
