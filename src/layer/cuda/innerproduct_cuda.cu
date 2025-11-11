#include <cuda_runtime.h>
#include "innerproduct_cuda.h"

namespace ncnn {
    __global__ void GEMM_kernel_float32(const void* input_blob, const void* weight_blob, void* output_blob, int M, int N, int K)
    {
        // 声明线程块共享内存变量
        constexpr int TILE_SIZE = 4;
        __shared__ float As[TILE_SIZE][TILE_SIZE];
        __shared__ float Bs[TILE_SIZE][TILE_SIZE];

        // 当前线程的全局行列索引（对应输出矩阵 C 的位置）
        unsigned int row = blockIdx.y * TILE_SIZE + threadIdx.y;    // 当前线程处理的行号
        unsigned int col = blockIdx.x * TILE_SIZE + threadIdx.x;    // 当前线程处理的列号

        // 将 void* 转换为 float* 方便计算
        const auto* A = static_cast<const float*>(input_blob);      // 输入矩阵 A
        const auto* B = static_cast<const float*>(weight_blob);     // 权重矩阵 B
        auto* C = static_cast<float*>(output_blob);                 // 输出矩阵 C


        // Tile Gemm算法（子块）
        float sum = 0.0f;
        for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++)
        {
            // 从全局内存加载 A 的一个 tile 到共享内存
            if (row < M && t * TILE_SIZE + threadIdx.x < K)
                As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
            else
                As[threadIdx.y][threadIdx.x] = 0.0f;

            // 从全局内存加载 B 的一个 tile 到共享内存
            if (col < N && t * TILE_SIZE + threadIdx.y < K)
                Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
            else
                Bs[threadIdx.y][threadIdx.x] = 0.0f;

            __syncthreads(); // 同步所有线程，确保 tile 都加载完

            // 在共享内存中进行部分乘加
            for (int k = 0; k < TILE_SIZE; k++)
                sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];

            __syncthreads(); // 等待所有线程完成该 tile 的计算
        }
        // 写回结果（只写有效范围内的 C）
        if (row < M && col < N)
            C[row * N + col] = sum;
    }

    __global__ void GEMM_kernel_half16(const void* input_blob, const void* weight_blob, void* output_blob, int M, int N, int K)
    {

    }

    __global__ void GEMM_kernel_int8(const void* input_blob, const void* weight_blob, void* output_blob, int M, int N, int K)
    {

    }

    int InnerProduct_cuda::InnerProduct_cuda_forward(const CudaMat& input_blob, CudaMat& output_blob, const Option& opt) const
    {
        // ========== 1. 准备输入 ==========
        int total_input_size = 0; // 展平后的输入长度
        if (input_blob.dims == 3)
        {
            total_input_size = input_blob.c * input_blob.h * input_blob.w; // 展平后总长度
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
        const int M = input_blob.h;          // 输入矩阵行数（batch size）
        const int K = input_blob.w;          // 输入矩阵列数（输入特征数）
        const int N = weight_blob.h;         // 权重矩阵的输出维度（输出特征数）

        // 根据数据精度类型选择对应的核函数
        if (input_blob.elemsize == 4)
        {
            dim3 blocksPerGrid(
                (N + 4 - 1) / 4,   // 横向方向：覆盖所有列
                (M + 4 - 1) / 4    // 纵向方向：覆盖所有行
            );
            dim3 threadsPerBlock(4, 4);
            GEMM_kernel_float32<<<blocksPerGrid, threadsPerBlock>>>(
                input_blob.data, weight_blob.data, output_blob.data, M, N, K);
        }
        else if (input_blob.elemsize == 2)
        {
            dim3 blocksPerGrid(
                (N + 8 - 1) / 8,   // 横向方向：覆盖所有列
                (M + 8 - 1) / 8    // 纵向方向：覆盖所有行
            );
            dim3 threadsPerBlock(8, 8);
            GEMM_kernel_half16<<<blocksPerGrid, threadsPerBlock>>>(
                input_blob.data, weight_blob.data, output_blob.data, M, N, K);
        }
        else if (input_blob.elemsize == 1)
        {
            dim3 blocksPerGrid(
                (N + 4 - 1) / 4,   // 横向方向：覆盖所有列
                (M + 4 - 1) / 4    // 纵向方向：覆盖所有行
            );
            dim3 threadsPerBlock(4, 4);
            GEMM_kernel_int8<<<blocksPerGrid, threadsPerBlock>>>(
                input_blob.data, weight_blob.data, output_blob.data, M, N, K);
        }

        // 3.应用激活函数

        // 4.同步设备，等待内核执行完成。这个函数会阻塞主机（CPU）直到设备（GPU）上的所有操作完成。
        cudaDeviceSynchronize();
        return 0;
    }
} // namespace ncnn