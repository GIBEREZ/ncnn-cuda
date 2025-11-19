#include "relu_cuda.h"
#include <cuda_runtime.h>

namespace ncnn {
    /**
     * CUDA版本激活函数kernel() 和 CUDA版本激活函数kernel的CPP API接口
     * @param input_blob 输入张量-线性数组指针
     * @param output_blob 输出张量-线性数组指针
     * @param number 线性数组元素个数
     * 在CUDA kernel里，x(也就是 const float* x)只是一个指向连续内存的线性数组指针，它本身不知道也不关心维度（shape）。
     */
    __global__ void relu_kernel_cuda(const float* input_blob, float* output_blob, int number)
    {
        // 计算全局线程索引（global thread index）
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int idxElement = idx * 4;
        if (idxElement + 3 < number) {
            float4 vec_in = *(float4*)&input_blob[idxElement];
            float4 vec_out;

            vec_out.x = vec_in.x > 0.0f ? vec_in.x : 0.0f;
            vec_out.y = vec_in.y > 0.0f ? vec_in.y : 0.0f;
            vec_out.z = vec_in.z > 0.0f ? vec_in.z : 0.0f;
            vec_out.w = vec_in.w > 0.0f ? vec_in.w : 0.0f;

            *(float4*)&output_blob[idxElement] = vec_out;
        }
        else {
            for (int i = 0; i < 4; i++) {
                unsigned int elem_idx = idxElement + i;
                if (elem_idx < number) {
                    output_blob[elem_idx] = input_blob[elem_idx] > 0.0f ? input_blob[elem_idx] : 0.0f;
                }
            }
        }
    }
    void relu_cuda(const CudaMat& input_blob, CudaMat& output_blob, int number)
    {
        int threadsPerBlock = 1024;                                                         // 定义每个线程块的线程数量
        int vec_size = 16 / input_blob.elemsize;                                            // 每个线程处理多少个元素（保证16B）
        int totalThreadsNeeded = (number + vec_size - 1) / vec_size;                        // 计算总共需要多少个线程来处理整个数组。
        int blocksPerGrid = (totalThreadsNeeded + threadsPerBlock - 1) / threadsPerBlock;   // 计算网格中线程块的数量。每个线程块有256个线程，所以总线程数除以每个线程块的线程数，并向上取整。

        relu_kernel_cuda<<<blocksPerGrid, threadsPerBlock>>>(static_cast<const float*>(input_blob.gpu_data), static_cast<float*>(output_blob.gpu_data), number);
        // 同步设备，等待内核执行完成。这个函数会阻塞主机（CPU）直到设备（GPU）上的所有操作完成。
        cudaDeviceSynchronize();
    }
}