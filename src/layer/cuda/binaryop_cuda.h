//
// Created by GIBEREZ on 2025/11/21.
//

#ifndef NCNN_BINARYOP_CUDA_H
#define NCNN_BINARYOP_CUDA_H

#include "layer.h"

namespace ncnn {
    enum OperationType
    {
        Operation_ADD = 0,
        Operation_SUB = 1,
        Operation_MUL = 2,
        Operation_DIV = 3,
        Operation_MAX = 4,
        Operation_MIN = 5,
        Operation_POW = 6,
        Operation_RSUB = 7,
        Operation_RDIV = 8,
        Operation_RPOW = 9,
        Operation_ATAN2 = 10,
        Operation_RATAN2 = 11
    };

    class BinaryOp_cuda : public Layer
    {
    public:
        BinaryOp_cuda();

        virtual int load_param(const ParamDict& pd);
        virtual int forward(const std::vector<CudaMat>& bottom_blobs, std::vector<CudaMat>& top_blobs, const Option& opt) const;
        virtual int forward_inplace(CudaMat& bottom_top_blob, const Option& opt) const;

        int binary_op_selector();

    public:
        int op_type;
        int with_scalar;
        float b;
    };
}

#endif //NCNN_BINARYOP_CUDA_H
