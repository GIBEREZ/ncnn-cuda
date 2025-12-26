//
// Created by GIBEREZ on 2025/11/21.
//

#include "binaryop_cuda.h"

namespace ncnn {
    BinaryOp_cuda::BinaryOp_cuda()
    {
        one_blob_only = false;
        support_inplace = false;
        support_cuda = true;
    }

    int BinaryOp_cuda::load_param(const ParamDict& pd)
    {
        op_type = pd.get(0, 0);
        with_scalar = pd.get(1, 0);
        b = pd.get(2, 0.f);

        if (with_scalar != 0)
        {
            one_blob_only = true;
            support_inplace = true;
        }

        return 0;
    }

    int BinaryOp_cuda::forward(const std::vector<CudaMat>& bottom_blobs, std::vector<CudaMat>& top_blobs, const Option& opt) const
    {
        NCNN_LOGE("  *  Running CUDA BinaryOp_%d forward", op_type);
        binaryop_cuda(bottom_blobs, top_blobs, opt);
        NCNN_LOGE("  *  forward output_blob w=%d,h=%d,d=%d,c=%d,dims=%d",output_blob.w,output_blob.h,output_blob.d,output_blob.c,output_blob.dims);
        if (output_blob.empty() || output_blob.gpu_data == nullptr) NCNN_LOGE("  *  output blob gpu_data == nullptr");
        NCNN_LOGE("  *  CUDA BinaryOp_%d forward done", op_type);
        return 0;
    }

    int BinaryOp_cuda::forward_inplace(CudaMat& bottom_top_blob, const Option& opt) const
    {

        return 0;
    }


}
