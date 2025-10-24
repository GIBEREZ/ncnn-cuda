//
// Created by GIBEREZ on 2025/10/21.
//

#ifndef NCNN_PADDING_CUDA_H
#define NCNN_PADDING_CUDA_H
#include "layer.h"

namespace ncnn {
    class Padding_cuda : public Layer
    {
    public:
        Padding_cuda();
        virtual int upload_model(const Option& opt);
        virtual int forward(const CudaMat& input_blob, CudaMat& output_blob, const Option& opt) const;
    };
}


#endif //NCNN_PADDING_CUDA_H
