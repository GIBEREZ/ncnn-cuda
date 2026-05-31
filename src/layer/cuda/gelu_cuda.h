//
// Created for ncnn-cuda
//

#ifndef NCNN_GELU_CUDA_H
#define NCNN_GELU_CUDA_H
#include "layer.h"

namespace ncnn {
int gelu_cuda(const CudaMat& input_blob, CudaMat& output_blob, bool fast_gelu);
int gelu_cuda_inplace(CudaMat& input_blob, bool fast_gelu);
    class GELU_cuda : public Layer
    {
    public:
        GELU_cuda();
        int load_param(const ParamDict& pd) override;
        using Layer::forward;
        int forward(const CudaMat& input_blob, CudaMat& output_blob, const Option& opt) const override;
        using Layer::forward_inplace;
        int forward_inplace(CudaMat& input_blob, const Option& opt) const override;
    public:
        bool fast_gelu;
    };
}

#endif //NCNN_GELU_CUDA_H
