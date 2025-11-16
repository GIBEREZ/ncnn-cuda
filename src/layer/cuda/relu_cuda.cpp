//
// Created by GIBEREZ on 2025/10/22.
//

#include "Relu_cuda.h"

#include <iostream>
#include <ostream>

namespace ncnn {
    ReLU_cuda::ReLU_cuda()
    {
        one_blob_only = true;
        support_inplace = true;
        support_cuda = true;
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
        NCNN_LOGE("  *  Running CUDA ReLU forward");
        relu_cuda(input_blob, output_blob, input_blob.total());
        NCNN_LOGE("  *  forward output_blob w=%d,h=%d,d=%d,c=%d,dims=%d",output_blob.w,output_blob.h,output_blob.d,output_blob.c,output_blob.dims);
        if (output_blob.empty() || output_blob.gpu_data == nullptr) NCNN_LOGE("  *  output blob gpu_data == nullptr");
        NCNN_LOGE("  *  CUDA ReLU forward done");
        return 0;
    }
} // namespace ncnn
