//
// Created by GIBEREZ on 2025/12/31.
//

#ifndef NCNN_ARGMAX_CUDA_H
#define NCNN_ARGMAX_CUDA_H
#include "layer.h"

namespace ncnn{
    class ArgMax_cuda : public Layer
    {
    public:
        ArgMax_cuda();

        int load_param(const ParamDict& pd) override;
        int forward(const CudaMat& input_blob, CudaMat& output_blob, const Option& opt) const override;
        int argmax_cuda(const CudaMat& input_blob, CudaMat& output_blob) const;
    public:
        int out_max_val;
        int topk;
    };


}

#endif //NCNN_ARGMAX_CUDA_H
