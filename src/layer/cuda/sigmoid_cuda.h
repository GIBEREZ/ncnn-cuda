//
// Created by GIBEREZ on 2025/12/26.
//

#ifndef NCNN_SIGMOID_CUDA_H
#define NCNN_SIGMOID_CUDA_H
#include "layer.h"

namespace ncnn {
    class Sigmoid_cuda : public Layer
    {
        public:
        Sigmoid_cuda();
        int forward_inplace(CudaMat& input_blob, const Option& opt) const override;
    };
}

#endif //NCNN_SIGMOID_CUDA_H
