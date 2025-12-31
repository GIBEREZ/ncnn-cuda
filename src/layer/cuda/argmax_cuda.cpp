//
// Created by GIBEREZ on 2025/12/31.
//

#include "argmax_cuda.h"

namespace ncnn{

    ArgMax_cuda::ArgMax_cuda()
    {
        one_blob_only = true;
        support_cuda = true;
    }

    int ArgMax_cuda::load_param(const ParamDict& pd)
    {
        out_max_val = pd.get(0, 0);
        topk = pd.get(1, 1);

        return 0;
    }

    int ArgMax_cuda::forward(const CudaMat& input_blob, CudaMat& output_blob, const Option& opt) const
    {
        NCNN_LOGE("  *  Running CUDA ArgMax forward");
        argmax_cuda(input_blob, output_blob);
        NCNN_LOGE("  *  forward output_blob w=%d,h=%d,d=%d,c=%d,dims=%d",output_blob.w,output_blob.h,output_blob.d,output_blob.c,output_blob.dims);
        if (output_blob.empty() || output_blob.gpu_data == nullptr) NCNN_LOGE("  *  output blob gpu_data == nullptr");
        NCNN_LOGE("  *  CUDA ArgMax forward done");
        return 0;
    }


}