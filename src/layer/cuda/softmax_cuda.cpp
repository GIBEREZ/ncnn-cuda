//
// Created by GIBEREZ on 2025/11/15.
//

#include "softmax_cuda.h"

namespace ncnn {
    Softmax_cuda::Softmax_cuda()
    {
        one_blob_only = true;
        support_inplace = true;
        support_cuda = true;
    }

    int Softmax_cuda::load_param(const ParamDict& pd)
    {
        return 0;
    }

    int Softmax_cuda::forward(const CudaMat& input_blob, CudaMat& output_blob) const
    {
        NCNN_LOGE("  *  Running CUDA Softmax forward");
        softmax_cuda(input_blob, output_blob);
        NCNN_LOGE("  *  forward output_blob w=%d,h=%d,d=%d,c=%d,dims=%d",output_blob.w,output_blob.h,output_blob.d,output_blob.c,output_blob.dims);
        if (output_blob.empty() || output_blob.gpu_data == nullptr) NCNN_LOGE("  *  output blob gpu_data == nullptr");
        NCNN_LOGE("  *  CUDA Softmax forward done");
        return 0;
    }

    int Softmax_cuda::forward_inplace(CudaMat& input_blob, const Option& opt) const
    {
        NCNN_LOGE("  *  Running CUDA Softmax forward");
        softmax_cuda_inplace(input_blob);
        NCNN_LOGE("  *  forward output_blob w=%d,h=%d,d=%d,c=%d,dims=%d",input_blob.w,input_blob.h,input_blob.d,input_blob.c,input_blob.dims);
        if (input_blob.empty() || input_blob.gpu_data == nullptr) NCNN_LOGE("  *  output blob gpu_data == nullptr");
        NCNN_LOGE("  *  CUDA Softmax forward done");
        return 0;
    }

}
