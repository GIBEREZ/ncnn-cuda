//
// Created by GIBEREZ on 2026/1/1.
//

#include "cumulativesum_cuda.h"

namespace ncnn {
    CumulativeSum_cuda::CumulativeSum_cuda()
    {
        one_blob_only = true;
        support_inplace = true;
        support_cuda = true;
    }

    int CumulativeSum_cuda::load_param(const ParamDict& pd)
    {
        axis = pd.get(0, 0);

        return 0;
    }

    int CumulativeSum_cuda::forward_inplace(CudaMat& input_blob, const Option& opt) const
    {
        NCNN_LOGE("  *  Running CUDA CumulativeSum forward  axis=%d", axis);
        cumulativesum_cuda_inplace(input_blob, axis);
        NCNN_LOGE("  *  forward output_blob w=%d,h=%d,d=%d,c=%d,dims=%d",input_blob.w,input_blob.h,input_blob.d,input_blob.c,input_blob.dims);
        if (input_blob.empty() || input_blob.gpu_data == nullptr) NCNN_LOGE("  *  output blob gpu_data == nullptr");
        NCNN_LOGE("  *  CUDA CumulativeSum forward done");
        return 0;
    }


}
