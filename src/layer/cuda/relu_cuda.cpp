//
// Created by GIBEREZ on 2025/10/22.
//

#include "Relu_cuda.h"

#include <iostream>
#include <ostream>

namespace ncnn {
    ReLU_cuda::ReLU_cuda()
    {
        // 是否该层只处理单个输入blob（张量）
        one_blob_only = true;
        // 是否支持原地计算
        support_inplace = true;
    }
    int ReLU_cuda::load_param(const ParamDict& pd)
    {
        slope = pd.get(0, 0.f);
        return 0;
    }

    int ReLU_cuda::upload_model(const Option& opt)
    {
        return 0;
    }

    int ReLU_cuda::forward(const CudaMat& input_blob, CudaMat& output_blob, const Option& opt) const
    {
        relu_cuda(input_blob, output_blob, input_blob.total());
        return 0;
    }
} // namespace ncnn
