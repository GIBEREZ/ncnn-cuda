//
// Created by GIBEREZ on 2025/11/21.
//

#include "cast_cuda.h"


namespace ncnn {
    Cast_cuda::Cast_cuda()
    {
        one_blob_only = true;
        support_cuda = true;
    }

    int Cast_cuda::load_param(const ParamDict& pd)
    {
        type_from = pd.get(0, 0);
        type_to = pd.get(1, 0);

        if (type_from == 4) type_from = 2;
        if (type_to == 4) type_to = 2;

        return 0;
    }

    int Cast_cuda::forward(const CudaMat& input_blob, CudaMat& output_blob, const Option& opt) const
    {
        NCNN_LOGE("  *  Running CUDA Cast forward");
        Cast_cuda_forward(input_blob, output_blob);
        NCNN_LOGE("  *  forward output_blob w=%d,h=%d,d=%d,c=%d,dims=%d",output_blob.w,output_blob.h,output_blob.d,output_blob.c,output_blob.dims);
        if (output_blob.empty() || output_blob.gpu_data == nullptr) NCNN_LOGE("  *  output blob gpu_data == nullptr");
        NCNN_LOGE("  *  CUDA Cast forward done");
        return 0;
    }

} // namespace ncnn