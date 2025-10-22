#include <cuda_runtime.h>
#include "relu_cuda.h"

/**
 * CUDA版本激活函数kernel() 和 CUDA版本激活函数kernel的CPP API接口
 * @param input_blob 输入张量-线性数组指针
 * @param output_blob 输出张量-线性数组指针
 * @param number 线性数组元素个数
 * 在CUDA kernel里，x(也就是 const float* x)只是一个指向连续内存的线性数组指针，它本身不知道也不关心维度（shape）。
 */
__global__ void relu_kernel(const float* input_blob, float* output_blob, int number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < number) output_blob[idx] = input_blob[idx] > 0.0f ? input_blob[idx] : 0.0f;
}
