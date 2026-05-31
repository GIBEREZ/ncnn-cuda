//
// Created by GIBEREZ on 2026/1/1.
//

#ifndef NCNN_CLIP_CUDA_H
#define NCNN_CLIP_CUDA_H
#include "layer.h"

namespace ncnn {
int clip_cuda(const CudaMat& input_blob, CudaMat& output_blob, float min, float max);
int clip_cuda_inplace(CudaMat& input_blob, float min, float max);
    class Clip_cuda : public Layer
    {
    public:
        Clip_cuda();

        int load_param(const ParamDict& pd);
        using Layer::forward;
        int forward(const CudaMat& input_blob, CudaMat& output_blob, const Option& opt) const;
        using Layer::forward_inplace;
        int forward_inplace(CudaMat& input_blob, const Option& opt) const;
    public:
        float min;
        float max;
    };
}

#endif //NCNN_CLIP_CUDA_H
