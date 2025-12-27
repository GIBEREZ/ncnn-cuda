//
// Created by GIBEREZ on 2025/12/26.
//
#include "gemm_cuda.h"
#include <cublas_v2.h>

namespace ncnn {
    int Gemm_cuda::gemm_cuda(const std::vector<CudaMat>& bottom_blobs, std::vector<CudaMat>& top_blobs, const Option& opt) const
    {
        // 1.确定输入矩阵，如果constant被设置，则使用从模型读取的矩阵，如果没有则采用bottom_blobs中的
        const CudaMat& A0 = constantA ? A : bottom_blobs[0];
        const CudaMat& B0 = constantB ? B : constantA ? bottom_blobs[0] : bottom_blobs[1];

        // 2.确定MKN，如果constant被设置，则使用从模型读取的值，如果没有则采用bottom_blobs中的
        const int M = constantM ? constantM : transA ? A0.w : (A0.dims == 3 ? A0.c : A0.h);
        const int K = constantK ? constantK : transA ? (A0.dims == 3 ? A0.c : A0.h) : A0.w;
        const int N = constantN ? constantN : transB ? (B0.dims == 3 ? B0.c : B0.h) : B0.w;

        CudaMat C0;
        if (constantA && constantB) C0 = bottom_blobs.size() == 1 ? bottom_blobs[0] : CudaMat();
        else if (constantA) C0 = bottom_blobs.size() == 2 ? bottom_blobs[1] : CudaMat();
        else if (constantB) C0 = bottom_blobs.size() == 2 ? bottom_blobs[1] : CudaMat();
        else C0 = bottom_blobs.size() == 3 ? bottom_blobs[2] : CudaMat();
        

        int elemsize = A0.elemsize;
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

        cublasHandle_t handle;
        cublasCreate(&handle);

        cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
        cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

        const int lda = (opA == CUBLAS_OP_N) ? K : M;
        const int ldb = (opB == CUBLAS_OP_N) ? N : K;
        const int ldc = output_transpose ? M : N;

        cublasGemmEx(
            handle,
            opA,opB,
            M, N, K,
            &alpha,
            A0.gpu_data, CUDA_R_32F, lda,
            B0.gpu_data, CUDA_R_32F, ldb,
            &beta,
            C.gpu_data, CUDA_R_32F, ldc,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT
            );

        cublasDestroy_v2(handle);

        return 0;
    }
} // namespace ncnn