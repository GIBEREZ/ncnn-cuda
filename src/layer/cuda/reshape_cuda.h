//
// Created by GIBEREZ on 2025/11/7.
//

#ifndef NCNN_RESHAPE_H
#define NCNN_RESHAPE_H
#include "layer.h"
#include "expression.h"
#include "reshape.h"

namespace ncnn {
    class Reshape_cuda : public Reshape
    {
    public:
        Reshape_cuda();
        virtual int load_param(const ParamDict& pd);
        using Layer::forward;
        virtual int forward(const std::vector<CudaMat>& input_blobs, std::vector<CudaMat>& output_blobs, const Option& opt) const;
    };
}


#endif //NCNN_RESHAPE_H
