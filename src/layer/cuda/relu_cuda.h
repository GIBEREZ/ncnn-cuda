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
        virtual int load_param(const ParamDict& pd);
        virtual int upload_model(const Option& opt);
        virtual int forward(const CudaMat& input_blob, CudaMat& output_blob, const Option& opt) const;
    };
}


#endif //NCNN_RELU_CUDA_H
