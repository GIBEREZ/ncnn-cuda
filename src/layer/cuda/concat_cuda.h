//
// Created by GIBEREZ on 2026/1/1.
//

#ifndef NCNN_CONCAT_CUDA_H
#define NCNN_CONCAT_CUDA_H
#include "layer.h"

namespace ncnn {
void Concat_dims2_axis1(const CudaMat& input_blob, CudaMat& output_blob, int h, int input_blob_index, int offset, int top_w);
    class Concat_cuda : public Layer
    {
    public:
        Concat_cuda();

        int load_param(const ParamDict& pd) override;
        int forward(const std::vector<CudaMat>& bottom_blobs, std::vector<CudaMat>& top_blobs, const Option& opt) const override;

    public:
        int axis;
    };
}

#endif //NCNN_CONCAT_CUDA_H
