// GELU CUDA kernel
//   exact: 0.5 * x * (1 + erf(x / sqrt(2)))
//   fast:  0.5 * x * (1 + tanh(sqrt(2/PI) * (x + 0.044715 * x^3)))

#include "gelu_cuda.h"

namespace ncnn {

// sqrt(2/PI) ≈ 0.7978845608
// sqrt(2)   ≈ 1.4142135624
// 0.044715 = constant for tanh approximation

__global__ void gelu_kernel_cuda(const float* input, float* output, int number, bool fast_gelu)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int idxElement = idx * 4;

    if (fast_gelu)
    {
        const float k = 0.7978845608f;  // sqrt(2/PI)
        const float c = 0.044715f;
        if (idxElement + 3 < number) {
            float4 vec_in = *(const float4*)&input[idxElement];
            float4 vec_out;
            vec_out.x = 0.5f * vec_in.x * (1.f + tanhf(k * (vec_in.x + c * vec_in.x * vec_in.x * vec_in.x)));
            vec_out.y = 0.5f * vec_in.y * (1.f + tanhf(k * (vec_in.y + c * vec_in.y * vec_in.y * vec_in.y)));
            vec_out.z = 0.5f * vec_in.z * (1.f + tanhf(k * (vec_in.z + c * vec_in.z * vec_in.z * vec_in.z)));
            vec_out.w = 0.5f * vec_in.w * (1.f + tanhf(k * (vec_in.w + c * vec_in.w * vec_in.w * vec_in.w)));
            *(float4*)&output[idxElement] = vec_out;
        } else {
            for (int i = 0; i < 4; i++) {
                unsigned int elem_idx = idxElement + i;
                if (elem_idx < number) {
                    float x = input[elem_idx];
                    output[elem_idx] = 0.5f * x * (1.f + tanhf(k * (x + c * x * x * x)));
                }
            }
        }
    }
    else  // exact: 0.5 * x * (1 + erf(x / sqrt(2)))
    {
        const float rsqrt2 = 0.7071067812f;  // 1/sqrt(2)
        if (idxElement + 3 < number) {
            float4 vec_in = *(const float4*)&input[idxElement];
            float4 vec_out;
            vec_out.x = 0.5f * vec_in.x * (1.f + erff(vec_in.x * rsqrt2));
            vec_out.y = 0.5f * vec_in.y * (1.f + erff(vec_in.y * rsqrt2));
            vec_out.z = 0.5f * vec_in.z * (1.f + erff(vec_in.z * rsqrt2));
            vec_out.w = 0.5f * vec_in.w * (1.f + erff(vec_in.w * rsqrt2));
            *(float4*)&output[idxElement] = vec_out;
        } else {
            for (int i = 0; i < 4; i++) {
                unsigned int elem_idx = idxElement + i;
                if (elem_idx < number) {
                    float x = input[elem_idx];
                    output[elem_idx] = 0.5f * x * (1.f + erff(x * rsqrt2));
                }
            }
        }
    }
}

int gelu_cuda(const CudaMat& input_blob, CudaMat& output_blob, bool fast_gelu)
{
    int number = input_blob.total();
    if (output_blob.empty())
        output_blob.create_like(input_blob);

    int threadsPerBlock = 256;
    int totalThreads = (number + 3) / 4;
    int blocksPerGrid = (totalThreads + threadsPerBlock - 1) / threadsPerBlock;

    gelu_kernel_cuda<<<blocksPerGrid, threadsPerBlock>>>(
        static_cast<const float*>(input_blob.gpu_data),
        static_cast<float*>(output_blob.gpu_data),
        number, fast_gelu);

    cudaDeviceSynchronize();
    return 0;
}

int gelu_cuda_inplace(CudaMat& input_blob, bool fast_gelu)
{
    int number = input_blob.total();
    int threadsPerBlock = 256;
    int totalThreads = (number + 3) / 4;
    int blocksPerGrid = (totalThreads + threadsPerBlock - 1) / threadsPerBlock;

    float* gpu_data = static_cast<float*>(input_blob.gpu_data);
    gelu_kernel_cuda<<<blocksPerGrid, threadsPerBlock>>>(
        gpu_data, gpu_data, number, fast_gelu);

    cudaDeviceSynchronize();
    return 0;
}

} // namespace ncnn
