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
        virtual int load_model(const ModelBin& mb);
        virtual int forward_inplace(CudaMat& output_blob, const Option& opt) const;
        int Bias_cuda_forward_inplace(CudaMat& input_blob) const;
    public:
        int bias_data_size;
        CudaMat bias_blob;
    };
}

#endif //NCNN_BIAS_H
