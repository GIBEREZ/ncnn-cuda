//
// Created by GIBEREZ on 2025/12/26.
//

#include "sigmoid_cuda.h"

namespace ncnn {
    Sigmoid_cuda::Sigmoid_cuda()
    {
        one_blob_only = true;
        support_inplace = true;
        support_cuda = true;
    }

    int Sigmoid_cuda::forward_inplace(CudaMat& input_blob, const Option& opt) const
    {

    }

}