//
// Created by GIBEREZ on 2025/11/7.
//

#include "reshape_cuda.h"

namespace ncnn {
    Reshape_cuda::Reshape_cuda()
    {
        one_blob_only = true;
        support_cuda = true;
    }

    int Reshape_cuda::load_param(const ParamDict& pd)
    {
        return 0;
    }

    int Reshape_cuda::load_model(const ModelBin& mb)
    {
        return 0;
    }

    int Reshape_cuda::forward(const CudaMat& input_blob, CudaMat& output_blob, const Option& opt) const
    {
        return 0;
    }

}
