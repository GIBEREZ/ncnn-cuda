#include "binaryop_cuda.h"
#include <cuda_fp16.h>

namespace ncnn {
    struct Operation_add
    {
        __host__ __device__ float operator() (const float& x, const float& y) const
        {
            return x + y;
        }

        __host__ __device__ half operator() (const half& x, const half& y) const
        {
            return x + y;
        }

    };

    struct Operation_sub
    {
        __host__ __device__ float operator() (const float& x, const float& y) const
        {
            return x - y;
        }

        __host__ __device__ half operator() (const half& x, const half& y) const
        {
            return x - y;
        }
    };

    struct Operation_mul
    {
        __host__ __device__ float operator() (const float& x, const float& y) const
        {
            return x * y;
        }

        __host__ __device__ half operator() (const half& x, const half& y) const
        {
            return x * y;
        }
    };

    struct Operation_div
    {
        __host__ __device__ float operator() (const float& x, const float& y) const
        {
            return x / y;
        }

        __host__ __device__ half operator() (const half& x, const half& y) const
        {
            return x / y;
        }
    };

    struct Operation_max
    {
        __host__ __device__ float operator() (const float& x, const float& y) const
        {
            return fmaxf(x, y);
        }

        __host__ __device__ half operator() (const half& x, const half& y) const
        {
            return __hmax(x, y);
        }
    };

    struct Operation_min
    {
        __host__ __device__ float operator() (const float& x, const float& y) const
        {
            return fminf(x, y);
        }

        __host__ __device__ half operator() (const half& x, const half& y) const
        {
            return __hmin(x, y);
        }
    };

    struct Operation_pow
    {
        __host__ __device__ float operator() (const float& x, const float& y) const
        {
            return powf(x, y);
        }

        __host__ __device__ half operator() (const half& x, const half& y) const
        {
            float xf = __half2float(x);
            float yf = __half2float(y);
            float result = powf(xf, yf);
            return __float2half(result);
        }
    };

    struct Operation_rsub
    {
        __host__ __device__ float operator() (const float& x, const float& y) const
        {
            return y - x;
        }

        __host__ __device__ half operator() (const half& x, const half& y) const
        {
            return y - x;
        }
    };

    struct Operation_rdiv
    {
        __host__ __device__ float operator() (const float& x, const float& y) const
        {
            return y / x;
        }

        __host__ __device__ half operator() (const half& x, const half& y) const
        {
            return y / x;
        }
    };

    struct Operation_rpow
    {
        __host__ __device__ float operator() (const float& x, const float& y) const
        {
            return powf(y, x);
        }

        __host__ __device__ half operator() (const half& x, const half& y) const
        {
            float xf = __half2float(x);
            float yf = __half2float(y);
            float result = powf(yf, xf);
            return __float2half(result);
        }
    };

    struct Operation_atan2
    {
        __host__ __device__ float operator() (const float& x, const float& y) const
        {
            return atan2f(x,y);
        }

        __host__ __device__ half operator() (const half& x, const half& y) const
        {
            float xf = __half2float(x);
            float yf = __half2float(y);
            float result = atan2f(xf, yf);
            return __float2half(result);
        }
    };

    struct Operation_ratan2
    {
        __host__ __device__ float operator() (const float& x, const float& y) const
        {
            return atan2f(y,x);
        }

        __host__ __device__ half operator() (const half& x, const half& y) const
        {
            float xf = __half2float(x);
            float yf = __half2float(y);
            float result = atan2f(yf, xf);
            return __float2half(result);
        }
    };

    template<typename Op>
    __global__ void binaryop_kernel_FP32(float* A, float* B, float* C, int A_number)
    {
        const Op op;
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int idxElement = idx * 4;
        if (idxElement + 3 <= A_number)
        {
            C[idxElement] = op(A[idxElement], B[idxElement]);
            C[idxElement+1] = op(A[idxElement+1], B[idxElement+1]);
            C[idxElement+2] = op(A[idxElement+2], B[idxElement+2]);
            C[idxElement+3] = op(A[idxElement+3], B[idxElement+3]);
        }
    }

    template<typename Op>
    __global__ void binaryop_kernel_FP16(half* A, half* B, half* C, int A_number)
    {
        const Op op;
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int idxElement = idx * 4;
        if (idxElement + 3 <= A_number)
        {
            C[idxElement] = op(A[idxElement], B[idxElement]);
            C[idxElement+1] = op(A[idxElement+1], B[idxElement+1]);
            C[idxElement+2] = op(A[idxElement+2], B[idxElement+2]);
            C[idxElement+3] = op(A[idxElement+3], B[idxElement+3]);
        }
    }


    int BinaryOp_cuda::binary_op_broadcast(const CudaMat& input_blob, void* B, void* C, int A_number) const
    {
        int threads = 256;
        int blocks = (A_number + input_blob.elemsize * threads - 1) / (input_blob.elemsize * threads);
        switch(op_type)
        {
            case Operation_ADD:
                if (input_blob.elemsize == 4)
                {
                    binaryop_kernel_FP32<Operation_add><<<blocks, threads>>>(
                        static_cast<float*>(input_blob.gpu_data),static_cast<float*>(B),static_cast<float*>(C),A_number);
                }
                else if (input_blob.elemsize == 2)
                {
                    binaryop_kernel_FP16<Operation_add><<<blocks, threads>>>(
                        static_cast<half*>(input_blob.gpu_data),static_cast<half*>(B),static_cast<half*>(C),A_number);
                }
                break;

            case Operation_SUB:
                if (input_blob.elemsize == 4)
                {
                    binaryop_kernel_FP32<Operation_sub><<<blocks, threads>>>(
                        static_cast<float*>(input_blob.gpu_data),static_cast<float*>(B),static_cast<float*>(C),A_number);
                }
                else if (input_blob.elemsize == 2)
                {
                    binaryop_kernel_FP16<Operation_sub><<<blocks, threads>>>(
                        static_cast<half*>(input_blob.gpu_data),static_cast<half*>(B),static_cast<half*>(C),A_number);
                }
                break;

            case Operation_MUL:
                if (input_blob.elemsize == 4)
                {
                    binaryop_kernel_FP32<Operation_mul><<<blocks, threads>>>(
                        static_cast<float*>(input_blob.gpu_data),static_cast<float*>(B),static_cast<float*>(C),A_number);
                }
                else if (input_blob.elemsize == 2)
                {
                    binaryop_kernel_FP16<Operation_mul><<<blocks, threads>>>(
                        static_cast<half*>(input_blob.gpu_data),static_cast<half*>(B),static_cast<half*>(C),A_number);
                }
                break;

            case Operation_DIV:
                if (input_blob.elemsize == 4)
                {
                    binaryop_kernel_FP32<Operation_div><<<blocks, threads>>>(
                        static_cast<float*>(input_blob.gpu_data),static_cast<float*>(B),static_cast<float*>(C),A_number);
                }
                else if (input_blob.elemsize == 2)
                {
                    binaryop_kernel_FP16<Operation_div><<<blocks, threads>>>(
                        static_cast<half*>(input_blob.gpu_data),static_cast<half*>(B),static_cast<half*>(C),A_number);
                }
                break;

            case Operation_MAX:
                if (input_blob.elemsize == 4)
                {
                    binaryop_kernel_FP32<Operation_max><<<blocks, threads>>>(
                        static_cast<float*>(input_blob.gpu_data),static_cast<float*>(B),static_cast<float*>(C),A_number);
                }
                else if (input_blob.elemsize == 2)
                {
                    binaryop_kernel_FP16<Operation_max><<<blocks, threads>>>(
                        static_cast<half*>(input_blob.gpu_data),static_cast<half*>(B),static_cast<half*>(C),A_number);
                }
                break;

            case Operation_MIN:
                if (input_blob.elemsize == 4)
                {
                    binaryop_kernel_FP32<Operation_min><<<blocks, threads>>>(
                        static_cast<float*>(input_blob.gpu_data),static_cast<float*>(B),static_cast<float*>(C),A_number);
                }
                else if (input_blob.elemsize == 2)
                {
                    binaryop_kernel_FP16<Operation_min><<<blocks, threads>>>(
                        static_cast<half*>(input_blob.gpu_data),static_cast<half*>(B),static_cast<half*>(C),A_number);
                }
                break;

            case Operation_POW:
                if (input_blob.elemsize == 4)
                {
                    binaryop_kernel_FP32<Operation_pow><<<blocks, threads>>>(
                        static_cast<float*>(input_blob.gpu_data),static_cast<float*>(B),static_cast<float*>(C),A_number);
                }
                else if (input_blob.elemsize == 2)
                {
                    binaryop_kernel_FP16<Operation_pow><<<blocks, threads>>>(
                        static_cast<half*>(input_blob.gpu_data),static_cast<half*>(B),static_cast<half*>(C),A_number);
                }
                break;

            case Operation_RSUB:
                if (input_blob.elemsize == 4)
                {
                    binaryop_kernel_FP32<Operation_rsub><<<blocks, threads>>>(
                        static_cast<float*>(input_blob.gpu_data),static_cast<float*>(B),static_cast<float*>(C),A_number);
                }
                else if (input_blob.elemsize == 2)
                {
                    binaryop_kernel_FP16<Operation_rsub><<<blocks, threads>>>(
                        static_cast<half*>(input_blob.gpu_data),static_cast<half*>(B),static_cast<half*>(C),A_number);
                }
                break;

            case Operation_RDIV:
                if (input_blob.elemsize == 4)
                {
                    binaryop_kernel_FP32<Operation_rdiv><<<blocks, threads>>>(
                        static_cast<float*>(input_blob.gpu_data),static_cast<float*>(B),static_cast<float*>(C),A_number);
                }
                else if (input_blob.elemsize == 2)
                {
                    binaryop_kernel_FP16<Operation_rdiv><<<blocks, threads>>>(
                        static_cast<half*>(input_blob.gpu_data),static_cast<half*>(B),static_cast<half*>(C),A_number);
                }
                break;

            case Operation_RPOW:
                if (input_blob.elemsize == 4)
                {
                    binaryop_kernel_FP32<Operation_rpow><<<blocks, threads>>>(
                        static_cast<float*>(input_blob.gpu_data),static_cast<float*>(B),static_cast<float*>(C),A_number);
                }
                else if (input_blob.elemsize == 2)
                {
                    binaryop_kernel_FP16<Operation_rpow><<<blocks, threads>>>(
                        static_cast<half*>(input_blob.gpu_data),static_cast<half*>(B),static_cast<half*>(C),A_number);
                }
                break;

            case Operation_ATAN2:
                if (input_blob.elemsize == 4)
                {
                    binaryop_kernel_FP32<Operation_atan2><<<blocks, threads>>>(
                        static_cast<float*>(input_blob.gpu_data),static_cast<float*>(B),static_cast<float*>(C),A_number);
                }
                else if (input_blob.elemsize == 2)
                {
                    binaryop_kernel_FP16<Operation_atan2><<<blocks, threads>>>(
                        static_cast<half*>(input_blob.gpu_data),static_cast<half*>(B),static_cast<half*>(C),A_number);
                }
                break;

            case Operation_RATAN2:
                if (input_blob.elemsize == 4)
                {
                    binaryop_kernel_FP32<Operation_ratan2><<<blocks, threads>>>(
                        static_cast<float*>(input_blob.gpu_data),static_cast<float*>(B),static_cast<float*>(C),A_number);
                }
                else if (input_blob.elemsize == 2)
                {
                    binaryop_kernel_FP16<Operation_ratan2><<<blocks, threads>>>(
                        static_cast<half*>(input_blob.gpu_data),static_cast<half*>(B),static_cast<half*>(C),A_number);
                }
                break;

            default:
                return -233;
        }

        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
        {
            NCNN_LOGE("CUDA kernel failed: %s", cudaGetErrorString(err));
            return -1;
        }

        return 0;
    }

    int BinaryOp_cuda::binary_op_broadcast_inplace(CudaMat& input_blob, void* B, int A_number)
    {
        return 0;
    }

    int BinaryOp_cuda::binaryop_cuda(const std::vector<CudaMat>& bottom_blobs, std::vector<CudaMat>& top_blobs) const
    {
        const CudaMat& A = bottom_blobs[0];
        const CudaMat& B = bottom_blobs[1];
        const int outdims = std::max(A.dims, B.dims);

        CudaMat A2 = A;
        CudaMat B2 = B;

        if (A.dims < outdims)
        {
            // expand inner axes
            if (outdims == 2)
            {
                if (A.w == B.h)
                    A2 = A.reshape(1, A.w);
                else // if (A.w == B.w)
                    A2 = A.reshape(A.w, 1);
            }
            if (outdims == 3 && A.dims == 1)
            {
                if (A.w == B.c)
                    A2 = A.reshape(1, 1, A.w);
                else // if (A.w == B.w)
                    A2 = A.reshape(A.w, 1, 1);
            }
            if (outdims == 3 && A.dims == 2)
                A2 = A.reshape(1, A.w, A.h);
            if (outdims == 4 && A.dims == 1)
            {
                if (A.w == B.c)
                    A2 = A.reshape(1, 1, 1, A.w);
                else // if (A.w == B.w)
                    A2 = A.reshape(A.w, 1, 1, 1);
            }
            if (outdims == 4 && A.dims == 2)
                A2 = A.reshape(1, 1, A.w, A.h);
            if (outdims == 4 && A.dims == 3)
                A2 = A.reshape(1, A.w, A.h, A.c);
        }
        if (B.dims < outdims)
        {
            // expand inner axes
            if (outdims == 2)
            {
                if (B.w == A.h)
                    B2 = B.reshape(1, B.w);
                else // if (B.w == A.w)
                    B2 = B.reshape(B.w, 1);
            }
            if (outdims == 3 && B.dims == 1)
            {
                if (B.w == A.c)
                    B2 = B.reshape(1, 1, B.w);
                else // if (B.w == A.w)
                    B2 = B.reshape(B.w, 1, 1);
            }
            if (outdims == 3 && B.dims == 2)
                B2 = B.reshape(1, B.w, B.h);
            if (outdims == 4 && B.dims == 1)
            {
                if (B.w == A.c)
                    B2 = B.reshape(1, 1, 1, B.w);
                else // if (B.w == A.w)
                    B2 = B.reshape(B.w, 1, 1, 1);
            }
            if (outdims == 4 && B.dims == 2)
                B2 = B.reshape(1, 1, B.w, B.h);
            if (outdims == 4 && B.dims == 3)
                B2 = B.reshape(1, B.w, B.h, B.c);
        }

        const int outw = std::max(A2.w, B2.w);
        const int outh = std::max(A2.h, B2.h);
        const int outd = std::max(A2.d, B2.d);
        const int outc = std::max(A2.c, B2.c);

        CudaMat& top_blob = top_blobs[0];
        if (outdims == 1)
        {
            top_blob.create(outw, 4u);
        }
        if (outdims == 2)
        {
            top_blob.create(outw, outh, 4u);
        }
        if (outdims == 3)
        {
            top_blob.create(outw, outh, outc, 4u);
        }
        if (outdims == 4)
        {
            top_blob.create(outw, outh, outd, outc, 4u);
        }
        if (top_blob.empty())
            return -100;

        binary_op_broadcast(A2, B2, top_blob, A.total());
        return 0;
    }
}