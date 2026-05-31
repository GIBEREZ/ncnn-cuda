//
// Created for ncnn-cuda
//

#ifndef NCNN_ELU_CUDA_H
#define NCNN_ELU_CUDA_H
#include "layer.h"

namespace ncnn {
int elu_cuda(const CudaMat& input_blob, CudaMat& output_blob, float alpha);
int elu_cuda_inplace(CudaMat& input_blob, float alpha);
    class ELU_cuda : public Layer
    {
    public:
        ELU_cuda();
        int load_param(const ParamDict& pd) override;
        using Layer::forward;
        int forward(const CudaMat& input_blob, CudaMat& output_blob, const Option& opt) const override;
        using Layer::forward_inplace;
        int forward_inplace(CudaMat& input_blob, const Option& opt) const override;
    public:
        float alpha;
    };
}

#endif //NCNN_ELU_CUDA_H
