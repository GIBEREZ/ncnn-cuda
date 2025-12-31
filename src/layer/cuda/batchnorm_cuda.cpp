//
// Created by GIBEREZ on 2025/12/31.
//

#include "batchnorm_cuda.h"

namespace ncnn {
    BatchNorm_cuda::BatchNorm_cuda()
    {
        one_blob_only = true;
        support_inplace = true;
        support_cuda = true;
    }

    int BatchNorm_cuda::load_param(const ParamDict& pd)
    {
        channels = pd.get(0, 0);
        eps = pd.get(1, 0.f);

        return 0;
    }

    int BatchNorm_cuda::load_model(const ModelBin& mb)
    {
        slope_data = CudaMat(mb.load(channels, 1));
        if (slope_data.empty())
            return 100;

        mean_data = CudaMat(mb.load(channels, 1));
        if (mean_data.empty())
            return -100;

        var_data = CudaMat(mb.load(channels, 1));
        if (var_data.empty())
            return -100;

        bias_data = CudaMat(mb.load(channels, 1));
        if (bias_data.empty())
            return -100;

        a_data.create(channels);
        if (a_data.empty())
            return -100;
        b_data.create(channels);
        if (b_data.empty())
            return -100;

        batchnorm_precompute();
        return 0;
    }

    int BatchNorm_cuda::forward_inplace(CudaMat& input_blob, const Option& opt) const
    {
        NCNN_LOGE("  *  Running CUDA BatchNorm forward");
        batchnorm_cuda_inplace(input_blob);
        NCNN_LOGE("  *  forward output_blob w=%d,h=%d,d=%d,c=%d,dims=%d",input_blob.w,input_blob.h,input_blob.d,input_blob.c,input_blob.dims);
        if (input_blob.empty() || input_blob.gpu_data == nullptr) NCNN_LOGE("  *  output blob gpu_data == nullptr");
        NCNN_LOGE("  *  CUDA BatchNorm forward done");
        return 0;
    }

}