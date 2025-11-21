//
// Created by GIBEREZ on 2025/11/21.
//

#include "bias_cuda.h"

namespace ncnn {

    Bias_cuda::Bias_cuda()
    {
        one_blob_only = true;
        support_inplace = true;
        support_cuda = true;
    }
    int Bias_cuda::load_param(const ParamDict& pd)
    {
        bias_data_size = pd.get(0, 0);
        return 0;
    }

    int Bias_cuda::load_model(const ModelBin& mb)
    {
        bias_blob = CudaMat(mb.load(bias_data_size, 0));
        if (bias_blob.empty())
        {
            NCNN_LOGE("===Bias_cuda::load_model(const ModelBin& mb)=== bias_blob.empty() failed");
            return -100;
        }
        return 0;
    }

    int Bias_cuda::forward_inplace(CudaMat& output_blob, const Option& opt) const
    {
        NCNN_LOGE("  *  Running CUDA Bias forward");
        Bias_cuda_forward_inplace(output_blob);
        NCNN_LOGE("  *  forward output_blob w=%d,h=%d,d=%d,c=%d,dims=%d",output_blob.w,output_blob.h,output_blob.d,output_blob.c,output_blob.dims);
        if (output_blob.empty() || output_blob.gpu_data == nullptr) NCNN_LOGE("  *  output blob gpu_data == nullptr");
        NCNN_LOGE("  *  CUDA Bias forward done");
        return 0;
    }
}