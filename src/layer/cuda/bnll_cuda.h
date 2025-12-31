//
// Created by GIBEREZ on 2026/1/1.
//

#ifndef NCNN_BNLL_CUDA_H
#define NCNN_BNLL_CUDA_H
#include "layer.h"

namespace ncnn{
int bnll_cuda_inplace(CudaMat& input_blob);
    class BNLL_cuda : public Layer
    {
    public:
        BNLL_cuda();

        int forward_inplace(CudaMat& input_blob, const Option& opt) const override;
    };
}

#endif //NCNN_BNLL_CUDA_H
