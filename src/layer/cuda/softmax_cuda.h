//
// Created by GIBEREZ on 2025/11/15.
//

#ifndef NCNN_SOFTMAX_H
#define NCNN_SOFTMAX_H
#include "layer.h"

namespace ncnn {
    class Softmax_cuda : public Layer
    {
    public:
        Softmax_cuda();
        int load_param(const ParamDict& pd) override;
        int forward_inplace(const CudaMat& input_blob, const Option& opt) const ;
    };
}

#endif //NCNN_SOFTMAX_H
