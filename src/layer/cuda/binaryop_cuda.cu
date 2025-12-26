#include "binaryop_cuda.h"
#include <cuda_fp16.h>

namespace ncnn {
    struct Operation_ADD
    {
        __global__ float operator() (const float& x, const float& y) const
        {
            return x + y;
        }

        __global__ half operator() (const half& x, const half& y) const
        {
            return x + y;
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


    int BinaryOp_cuda::binary_op_selector()
    {
        if (op_type == Operation_ADD) return <Operation_ADD>(a, b, c, opt);;
        if (op_type == Operation_SUB) return 0;
        if (op_type == Operation_MUL) return 0;
        if (op_type == Operation_DIV) return 0;
        if (op_type == Operation_MAX) return 0;
        if (op_type == Operation_MIN) return 0;
        if (op_type == Operation_POW) return 0;
        if (op_type == Operation_RSUB) return 0;
        if (op_type == Operation_RDIV) return 0;
        if (op_type == Operation_RPOW) return 0;
        if (op_type == Operation_ATAN2) return 0;
        if (op_type == Operation_RATAN2) return 0;
    }

}