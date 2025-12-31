//
// Created by GIBEREZ on 2026/1/1.
//

#include "celu_cuda.h"
namespace ncnn {
    CELU_cuda::CELU_cuda()
    {
        one_blob_only = true;
        support_inplace = true;
        support_cuda = true;
    }

    int CELU_cuda::load_param(const ParamDict& pd)
    {
        alpha = pd.get(0, 1.f);

        return 0;
    }

    int CELU_cuda::forward_inplace(CudaMat& input_blob, const Option& opt) const
    {
        NCNN_LOGE("  *  Running CUDA CELU forward");
        celu_cuda_inplace(input_blob, alpha);
        NCNN_LOGE("  *  forward output_blob w=%d,h=%d,d=%d,c=%d,dims=%d",input_blob.w,input_blob.h,input_blob.d,input_blob.c,input_blob.dims);
        if (input_blob.empty() || input_blob.gpu_data == nullptr) NCNN_LOGE("  *  output blob gpu_data == nullptr");
        NCNN_LOGE("  *  CUDA CELU forward done");
        return 0;
    }
}