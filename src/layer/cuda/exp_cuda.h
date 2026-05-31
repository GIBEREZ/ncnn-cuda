//
// Created for ncnn-cuda
//

#ifndef NCNN_EXP_CUDA_H
#define NCNN_EXP_CUDA_H
#include "layer.h"

namespace ncnn {
int exp_cuda(const CudaMat& input_blob, CudaMat& output_blob, float base, float scale, float shift);
int exp_cuda_inplace(CudaMat& input_blob, float base, float scale, float shift);
    class Exp_cuda : public Layer
    {
    public:
        Exp_cuda();
        int load_param(const ParamDict& pd) override;
        using Layer::forward;
        int forward(const CudaMat& input_blob, CudaMat& output_blob, const Option& opt) const override;
        using Layer::forward_inplace;
        int forward_inplace(CudaMat& input_blob, const Option& opt) const override;
    public:
        float base;
        float scale;
        float shift;
    };
}

#endif //NCNN_EXP_CUDA_H
