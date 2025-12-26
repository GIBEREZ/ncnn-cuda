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
        binaryop_cuda(bottom_blobs, top_blobs);
        NCNN_LOGE("  *  forward output_blob w=%d,h=%d,d=%d,c=%d,dims=%d",top_blobs[0].w,top_blobs[0].h,top_blobs[0].d,top_blobs[0].c,top_blobs[0].dims);
        if (top_blobs[0].empty() || top_blobs[0].gpu_data == nullptr) NCNN_LOGE("  *  output blob gpu_data == nullptr");
        NCNN_LOGE("  *  CUDA BinaryOp_%d forward done", op_type);
        return 0;
    }

    int BinaryOp_cuda::forward_inplace(CudaMat& bottom_top_blob, const Option& opt) const
    {

        return 0;
    }


}
