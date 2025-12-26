#include <cuda_runtime.h>
#include "innerproduct_cuda.h"
#include "system.h"

namespace ncnn {
    __global__ void GEMM_kernel_FP32(const void* input_blob, const void* weight_blob, void* output_blob, int M, int N, int K)
    {
        constexpr int TILE_SIZE = 4;
        __shared__ float As[TILE_SIZE][TILE_SIZE];
        __shared__ float Bs[TILE_SIZE][TILE_SIZE];

        unsigned int row = blockIdx.y * TILE_SIZE + threadIdx.y;
        unsigned int col = blockIdx.x * TILE_SIZE + threadIdx.x;

        const auto* A = static_cast<const float*>(input_blob);
        const auto* B = static_cast<const float*>(weight_blob);
        auto* C = static_cast<float*>(output_blob);

        float sum = 0.0f;
        for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++)
        {
            if (row < M && t * TILE_SIZE + threadIdx.x < K)
                As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
            else
                As[threadIdx.y][threadIdx.x] = 0.0f;

            if (col < N && t * TILE_SIZE + threadIdx.y < K)
                Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
            else
                Bs[threadIdx.y][threadIdx.x] = 0.0f;

            __syncthreads();

            for (int k = 0; k < TILE_SIZE; k++)
                sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];

            __syncthreads();
        }

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
        const int M = input_blob.h;
        const int K = input_blob.w;
        const int N = num_output;

        if (output_blob.empty() || output_blob.w != N || output_blob.h != M || output_blob.dims != 2)
        {
            output_blob.release();
            output_blob.create(N, M, input_blob.elemsize);
            if (output_blob.empty())
            {
                NCNN_LOGE("===CUDA InnerProduct forward=== failed: output_blob allocation failed");
                return -100;
            }
        }

        if (input_blob.elemsize == 4)
        {
            dim3 blocksPerGrid(
                (N + 4 - 1) / 4,
                (M + 4 - 1) / 4
            );
            dim3 threadsPerBlock(4, 4);
            GEMM_kernel_FP32<<<blocksPerGrid, threadsPerBlock>>>(
                input_blob.gpu_data, weight_blob.gpu_data, output_blob.gpu_data, M, N, K);
        }
        else if (input_blob.elemsize == 2)
        {
            dim3 blocksPerGrid(
                (N + 8 - 1) / 8,
                (M + 8 - 1) / 8
            );
            dim3 threadsPerBlock(8, 8);
            GEMM_kernel_FP16<<<blocksPerGrid, threadsPerBlock>>>(
                input_blob.gpu_data, weight_blob.gpu_data, output_blob.gpu_data, M, N, K);
        }
        else if (input_blob.elemsize == 1)
        {
            dim3 blocksPerGrid(
                (N + 4 - 1) / 4,
                (M + 4 - 1) / 4
            );
            dim3 threadsPerBlock(4, 4);
            GEMM_kernel_INT8<<<blocksPerGrid, threadsPerBlock>>>(
                input_blob.gpu_data, weight_blob.gpu_data, output_blob.gpu_data, M, N, K);
        }

        cudaDeviceSynchronize();
        return 0;
    }
} // namespace ncnn