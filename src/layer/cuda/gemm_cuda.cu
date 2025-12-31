//
// Created by GIBEREZ on 2025/12/26.
//
#include "gemm_cuda.h"
#include <cublas_v2.h>

namespace ncnn {
    __global__ void broadcast_C_kernel(const float* input, float* output, int M, int N, int total, float alpha, int broadcast_type_C)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= total) return;

        const int i = idx / N;
        const int j = idx % N;

        if (broadcast_type_C == 0) output[idx] = input[0] * alpha;
        if (broadcast_type_C == 1 || broadcast_type_C == 2) output[idx] = input[i] * alpha;
        if (broadcast_type_C == 3) output[idx] = input[i * N + j] * alpha;
        if (broadcast_type_C == 4) output[idx] = input[j] * alpha;
    }

    int broadcast(const CudaMat& input, CudaMat& output, int M, int N, float alpha, int broadcast_type_C)
    {
        output.create(M, N);
        const int total = M * N;
        const int threads = 256;
        const int blocks = (total + threads - 1) / threads;

        broadcast_C_kernel<<<blocks, threads>>>(
            static_cast<const float*>(input.gpu_data), static_cast<float*>(output.gpu_data), M, N, total, alpha, broadcast_type_C);

        return 0;
    }

    int Gemm_cuda::gemm_cuda(const std::vector<CudaMat>& bottom_blobs, std::vector<CudaMat>& top_blobs, const Option& opt) const
    {
        const CudaMat& A = constantA ? model_A : bottom_blobs[0];
        const CudaMat& B = constantB ? model_B : constantA ? bottom_blobs[0] : bottom_blobs[1];

        const int M = constantM ? constantM : transA ? A.w : (A.dims == 3 ? A.c : A.h);
        const int K = constantK ? constantK : transA ? (A.dims == 3 ? A.c : A.h) : A.w;
        const int N = constantN ? constantN : transB ? (B.dims == 3 ? B.c : B.h) : B.w;

        int elemsize = A.elemsize;
        CudaMat& top_blob = top_blobs[0];
        if (output_transpose)
        {
            if (output_N1M) top_blob.create(M, 1, N, elemsize);
            else top_blob.create(M, N, elemsize);
        }
        else
        {
            if (output_N1M) top_blob.create(N, 1, M, elemsize);
            else top_blob.create(N, M, elemsize);
        }
        if (top_blob.empty()) return -100;

        CudaMat C;
        int broadcast_type_C = -1;
        if (constantC && constant_broadcast_type_C != -1)
        {
            // 浅拷贝该矩阵
            C = model_C;
            broadcast_type_C = constant_broadcast_type_C;
        }
        else
        {
            if (constantA && constantB) C = bottom_blobs.size() == 1 ? bottom_blobs[0] : CudaMat();
            else if (constantA) C = bottom_blobs.size() == 2 ? bottom_blobs[1] : CudaMat();
            else if (constantB) C = bottom_blobs.size() == 2 ? bottom_blobs[1] : CudaMat();
            else C = bottom_blobs.size() == 3 ? bottom_blobs[2] : CudaMat();
            if (!C.empty())
            {
                // scalar
                if (C.dims == 1 && C.w == 1) broadcast_type_C = 0;
                // M
                // auto broadcast from h to w is the ncnn-style convention
                if (C.dims == 1 && C.w == M) broadcast_type_C = 1;
                // N
                if (C.dims == 1 && C.w == N) broadcast_type_C = 4;
                // Mx1
                if (C.dims == 2 && C.w == 1 && C.h == M) broadcast_type_C = 2;
                // MxN
                if (C.dims == 2 && C.w == N && C.h == M) broadcast_type_C = 3;
                // 1xN
                if (C.dims == 2 && C.w == N && C.h == 1) broadcast_type_C = 4;
            }
        }

        broadcast(C, top_blob, M, N, alpha, broadcast_type_C);

        cublasHandle_t handle;
        cublasStatus_t status = cublasCreate(&handle);
        if (status != CUBLAS_STATUS_SUCCESS) return -233;

        cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
        cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

        const int lda = (opA == CUBLAS_OP_N) ? M : K;
        const int ldb = (opB == CUBLAS_OP_N) ? K : N;
        const int ldc = M;

        cublasGemmEx(
            handle,
            opA,opB,
            M, N, K,
            &alpha,
            A.gpu_data, CUDA_R_32F, lda,
            B.gpu_data, CUDA_R_32F, ldb,
            &beta,
            top_blob.gpu_data, CUDA_R_32F, ldc,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT
            );

        cublasDestroy(handle);

        return 0;
    }
} // namespace ncnn