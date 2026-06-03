//
// Created by GIBEREZ on 2026/1/1.
//

#ifndef NCNN_CUMULATIVESUM_CUDA_H
#define NCNN_CUMULATIVESUM_CUDA_H
#include "layer.h"

namespace ncnn {
int cumulativesum_cuda_inplace(CudaMat& input_blob, int axis);
    class CumulativeSum_cuda : public Layer
    {
    public:
        CumulativeSum_cuda();
        int load_param(const ParamDict& pd) override;
        using Layer::forward_inplace;
        int forward_inplace(CudaMat& input_blob, const Option& opt) const override;
    public:
        int axis;
    };
}

#endif //NCNN_CUMULATIVESUM_CUDA_H
