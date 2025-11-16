//
// Created by GIBEREZ on 2025/11/15.
//

#include "softmax_cuda.h"

namespace ncnn {
    Softmax_cuda::Softmax_cuda()
    {

    }

    int Softmax_cuda::load_param(const ParamDict& pd)
    {
        return 0;
    }

    int Softmax_cuda::forward_inplace(const CudaMat& input_blob, const Option& opt) const
    {
        return 0;
    }

}
