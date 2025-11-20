//
// Created by GIBEREZ on 2025/11/21.
//

#ifndef NCNN_BIAS_H
#define NCNN_BIAS_H

#include "layer.h"

namespace ncnn {
    class Bias_cuda : public Layer
    {
    public:
        Bias_cuda();
        virtual int load_param(const ParamDict& pd);
        virtual int forward(const CudaMat& input_blob, CudaMat& output_blob, const Option& opt) const;
        virtual int forward_inplace(CudaMat& output_blob, const Option& opt) const;
    public:
        // param
        int bias_data_size;
        // model
        CudaMat bias_data;
    };
}

#endif //NCNN_BIAS_H
