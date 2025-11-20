//
// Created by GIBEREZ on 2025/11/21.
//

#ifndef NCNN_CAST_H
#define NCNN_CAST_H
#include "layer.h"

namespace ncnn {
    class Cast_cuda : public Layer
    {
    public:
        Cast_cuda();
        virtual int load_param(const ParamDict& pd);
        virtual int forward(const CudaMat& bottom_blob, CudaMat& top_blob, const Option& opt) const;
        int Cast_cuda_forward(const CudaMat& input_blob, CudaMat& output_blob) const;
    public:
        // CUDA element type
        // 0 = auto
        // 1 = float32
        // 2 = half16
        // 3 = int8
        int type_from;
        int type_to;
    };
}

#endif //NCNN_CAST_H
