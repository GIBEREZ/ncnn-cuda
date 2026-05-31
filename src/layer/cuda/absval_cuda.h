//
// Created by GIBEREZ on 2025/12/31.
//

#ifndef NCNN_ABSVAL_CUDA_H
#define NCNN_ABSVAL_CUDA_H
#include "layer.h"

namespace ncnn {
void absval_cuda_inplace(CudaMat& input_blob);
    class AbsVal_cuda : public Layer
    {
        public:
        AbsVal_cuda();
        using Layer::forward_inplace;
        int forward_inplace(CudaMat& input_blob, const Option& opt) const override;
    };
}

#endif //NCNN_ABSVAL_CUDA_H
