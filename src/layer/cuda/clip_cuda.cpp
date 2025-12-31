//
// Created by GIBEREZ on 2026/1/1.
//

#include "clip_cuda.h"

namespace ncnn{
    Clip_cuda::Clip_cuda()
    {
        one_blob_only = true;
        support_inplace = true;
        support_cuda = true;
    }

    int Clip_cuda::load_param(const ParamDict& pd)
    {
        min = pd.get(0, -FLT_MAX);
        max = pd.get(1, FLT_MAX);

        return 0;
    }

    int Clip_cuda::forward_inplace(CudaMat& input_blob, const Option& opt) const
    {
        NCNN_LOGE("  *  Running CUDA Clip forward");
        clip_cuda_inplace(input_blob, min, max);
        NCNN_LOGE("  *  forward output_blob w=%d,h=%d,d=%d,c=%d,dims=%d",input_blob.w,input_blob.h,input_blob.d,input_blob.c,input_blob.dims);
        if (input_blob.empty() || input_blob.gpu_data == nullptr) NCNN_LOGE("  *  output blob gpu_data == nullptr");
        NCNN_LOGE("  *  CUDA Clip forward done");
        return 0;
    }


}