// Exp CUDA kernel: output = base ^ (scale * x + shift)
//   base = -1 → exp(scale * x + shift)
//   base > 0  → pow(base, scale * x + shift)

#include "exp_cuda.h"
#include <float.h>

namespace ncnn {

__global__ void exp_kernel_cuda(const float* input, float* output, int number,
                                 float base, float scale, float shift)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int idxElement = idx * 4;

    if (base <= 0.f)  // pure exp: exp(scale * x + shift)
    {
        if (idxElement + 3 < number) {
            float4 vec_in = *(const float4*)&input[idxElement];
            float4 vec_out;
            vec_out.x = expf(scale * vec_in.x + shift);
            vec_out.y = expf(scale * vec_in.y + shift);
            vec_out.z = expf(scale * vec_in.z + shift);
            vec_out.w = expf(scale * vec_in.w + shift);
            *(float4*)&output[idxElement] = vec_out;
        } else {
            for (int i = 0; i < 4; i++) {
                unsigned int elem_idx = idxElement + i;
                if (elem_idx < number)
                    output[elem_idx] = expf(scale * input[elem_idx] + shift);
            }
        }
    }
    else  // pow: base ^ (scale * x + shift) = exp((scale*x+shift) * log(base))
    {
        float log_base = logf(base);
        if (idxElement + 3 < number) {
            float4 vec_in = *(const float4*)&input[idxElement];
            float4 vec_out;
            vec_out.x = expf((scale * vec_in.x + shift) * log_base);
            vec_out.y = expf((scale * vec_in.y + shift) * log_base);
            vec_out.z = expf((scale * vec_in.z + shift) * log_base);
            vec_out.w = expf((scale * vec_in.w + shift) * log_base);
            *(float4*)&output[idxElement] = vec_out;
        } else {
            for (int i = 0; i < 4; i++) {
                unsigned int elem_idx = idxElement + i;
                if (elem_idx < number)
                    output[elem_idx] = expf((scale * input[elem_idx] + shift) * log_base);
            }
        }
    }
}

int exp_cuda(const CudaMat& input_blob, CudaMat& output_blob, float base, float scale, float shift)
{
    int number = input_blob.total();
    if (output_blob.empty())
        output_blob.create_like(input_blob);

    int threadsPerBlock = 256;
    int totalThreads = (number + 3) / 4;
    int blocksPerGrid = (totalThreads + threadsPerBlock - 1) / threadsPerBlock;

    exp_kernel_cuda<<<blocksPerGrid, threadsPerBlock>>>(
        static_cast<const float*>(input_blob.gpu_data),
        static_cast<float*>(output_blob.gpu_data),
        number, base, scale, shift);

    cudaDeviceSynchronize();
    return 0;
}

int exp_cuda_inplace(CudaMat& input_blob, float base, float scale, float shift)
{
    int number = input_blob.total();
    int threadsPerBlock = 256;
    int totalThreads = (number + 3) / 4;
    int blocksPerGrid = (totalThreads + threadsPerBlock - 1) / threadsPerBlock;

    float* gpu_data = static_cast<float*>(input_blob.gpu_data);
    exp_kernel_cuda<<<blocksPerGrid, threadsPerBlock>>>(
        gpu_data, gpu_data, number, base, scale, shift);

    cudaDeviceSynchronize();
    return 0;
}

} // namespace ncnn
