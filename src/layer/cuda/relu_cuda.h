//
// Created by GIBEREZ on 2025/10/22.
//

#ifndef NCNN_RELU_CUDA_H
#define NCNN_RELU_CUDA_H
#include "layer.h"

namespace ncnn {
void relu_cuda(const float* input_blob, float* output_blob, int number);
    class ReLU_cuda : public Layer
    {
    public:
        ReLU_cuda();
        int load_param(const ParamDict& pd) override;
        int upload_model(const Option& opt) override;
        int forward(const CudaMat& input_blob, CudaMat& output_blob, const Option& opt) const override;

        // 负斜率参数
        float slope;
    };

}

#endif //NCNN_RELU_CUDA_H
