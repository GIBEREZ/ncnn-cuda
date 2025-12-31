//
// Created by GIBEREZ on 2026/1/1.
//

#ifndef NCNN_CELU_CUDA_H
#define NCNN_CELU_CUDA_H
#include "layer.h"

namespace ncnn{
int celu_cuda_inplace(CudaMat& input_blob, float alpha);
    class CELU_cuda : public Layer
    {
    public:
        CELU_cuda();

        int load_param(const ParamDict& pd) override;
        int forward_inplace(CudaMat& input_blob, const Option& opt) const override;

    public:
        float alpha;
    };
}

#endif //NCNN_CELU_CUDA_H
