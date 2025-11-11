//
// Created by GIBEREZ on 2025/11/7.
//

#ifndef NCNN_RESHAPE_H
#define NCNN_RESHAPE_H
#include "layer.h"

namespace ncnn {
    class Reshape_cuda : public Layer
    {
    public:
        Reshape_cuda();
        virtual int load_param(const ParamDict& pd);
        virtual int load_model(const ModelBin& mb);
        virtual int forward(const CudaMat& input_blob, CudaMat& output_blob, const Option& opt) const;
    };
}


#endif //NCNN_RESHAPE_H
