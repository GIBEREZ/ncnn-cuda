//
// Created by GIBEREZ on 2025/11/15.
//

#ifndef NCNN_SOFTMAX_H
#define NCNN_SOFTMAX_H
#include "layer.h"

namespace ncnn {
void softmax_cuda(const CudaMat& input_blob, CudaMat& output_blob);
void softmax_cuda_inplace(CudaMat& input_blob);
    class Softmax_cuda : public Layer
    {
    public:
        Softmax_cuda();
        int load_param(const ParamDict& pd) override;
        int forward(const CudaMat& input_blob, CudaMat& output_blob) const;
        int forward_inplace(CudaMat& input_blob, const Option& opt) const;
    };
}

#endif //NCNN_SOFTMAX_H
