//
// Created by GIBEREZ on 2025/10/22.
//

#include "Relu_cuda.h"

namespace ncnn {
    ReLU_cuda::ReLU_cuda()
    {

    }
    int ReLU_cuda::load_param(const ParamDict& pd)
    {
        return 0;
    }

    int ReLU_cuda::upload_model(const Option& opt)
    {
        return 0;
    }

    int ReLU_cuda::forward(const CudaMat& input_blob, CudaMat& output_blob, const Option& opt) const
    {
        relu_kernel_cuda(input_blob, output_blob, input_blob.total());
        return 0;
    }
    } // namespace ncnn
