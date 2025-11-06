#include <cuda_runtime.h>
#include "innerproduct_cuda.h"

namespace ncnn {
    __global__ void GEMM_kernel_float32(const void* input_blob, const void* weight_blob, void* output_blob, int number)
    {
        // 计算全局线程索引（global thread index）
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x * 4;
        unsigned int idxElement = idx * 4;
        if (idxElement + 3 < number)
        {
            float4 vec_in = *(float4*)&input_blob[idxElement];
            float4 vec_out;
        }
    }

    __global__ void GEMM_kernel_half16(const void* input_blob, const void* weight_blob, void* output_blob, int number)
    {

    }

    __global__ void GEMM_kernel_int8(const void* input_blob, const void* weight_blob, void* output_blob, int number)
    {

    }

    int InnerProduct_cuda::InnerProduct_cuda_forward(const CudaMat& input_blob, CudaMat& output_blob, const Option& opt) const
    {
        // ========== 1. 准备输入 ==========
        int total_input_size = 0; // 展平后的输入长度
        if (input_blob.dims == 3)
        {
            const int channels = input_blob.c;  // 通道数
            const int h = input_blob.h;         // 高度
            const int w = input_blob.w;         // 宽度
            total_input_size = channels * h * w; // 展平后总长度
        }
        else if (input_blob.dims == 2)
        {
            total_input_size = input_blob.w * input_blob.h;
        }
        else if (input_blob.dims == 1)
        {
            total_input_size = input_blob.w;
        }
        else
        {
            NCNN_LOGE("Unsupported input dims: %d\n", input_blob.dims);
            return -100;
        }

        // 2.矩阵乘法 + 偏置项向量
        int threadsPerBlock = 1024;
        int totalThreadsNeeded = (total_input_size + 4 - 1) / 4;
        int blocksPerGrid = (totalThreadsNeeded + threadsPerBlock - 1) / threadsPerBlock;

        if (input_blob.elemsize == 4)
        {
            GEMM_kernel_float32<<<blocksPerGrid, threadsPerBlock>>>(input_blob.data, output_blob.data, number);
        }
        else if (input_blob.elemsize == 2)
        {
            GEMM_kernel_half16<<<blocksPerGrid, threadsPerBlock>>>(input_blob.data, output_blob.data, number);
        }
        else if (input_blob.elemsize == 1)
        {
            GEMM_kernel_int8<<<blocksPerGrid, threadsPerBlock>>>(input_blob.data, output_blob.data, number);
        }

        // 3.应用激活函数
        return 0;
    }
} // namespace ncnn