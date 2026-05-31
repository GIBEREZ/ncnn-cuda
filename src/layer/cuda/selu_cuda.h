//
// Created for ncnn-cuda
//

#ifndef NCNN_SELU_CUDA_H
#define NCNN_SELU_CUDA_H
#include "layer.h"

namespace ncnn {
int selu_cuda(const CudaMat& input_blob, CudaMat& output_blob, float alpha, float lambda);
int selu_cuda_inplace(CudaMat& input_blob, float alpha, float lambda);
    class SELU_cuda : public Layer
    {
    public:
        SELU_cuda();
        int load_param(const ParamDict& pd) override;
        using Layer::forward;
        int forward(const CudaMat& input_blob, CudaMat& output_blob, const Option& opt) const override;
        using Layer::forward_inplace;
        int forward_inplace(CudaMat& input_blob, const Option& opt) const override;
    public:
        float alpha;
        float lambda;
    };
}

#endif //NCNN_SELU_CUDA_H
