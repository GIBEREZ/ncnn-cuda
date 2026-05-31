//
// HardSwish CUDA layer implementation
//

#include "hardswish_cuda.h"

namespace ncnn {
    HardSwish_cuda::HardSwish_cuda()
    {
        one_blob_only = true;
        support_inplace = true;
        support_cuda = true;
    }

    int HardSwish_cuda::load_param(const ParamDict& pd)
    {
        alpha = pd.get(0, 0.2f);
        beta  = pd.get(1, 0.5f);
        return 0;
    }

    int HardSwish_cuda::forward(const CudaMat& input_blob, CudaMat& output_blob, const Option& opt) const
    {
        NCNN_LOGE("  *  Running CUDA HardSwish forward  alpha=%.4f beta=%.4f", alpha, beta);
        hardswish_cuda(input_blob, output_blob, alpha, beta);
        NCNN_LOGE("  *  forward output_blob w=%d,h=%d,d=%d,c=%d,dims=%d",output_blob.w,output_blob.h,output_blob.d,output_blob.c,output_blob.dims);
        NCNN_LOGE("  *  CUDA HardSwish forward done");
        return 0;
    }

    int HardSwish_cuda::forward_inplace(CudaMat& input_blob, const Option& opt) const
    {
        NCNN_LOGE("  *  Running CUDA HardSwish forward_inplace  alpha=%.4f beta=%.4f", alpha, beta);
        hardswish_cuda_inplace(input_blob, alpha, beta);
        NCNN_LOGE("  *  forward output_blob w=%d,h=%d,d=%d,c=%d,dims=%d",input_blob.w,input_blob.h,input_blob.d,input_blob.c,input_blob.dims);
        NCNN_LOGE("  *  CUDA HardSwish forward_inplace done");
        return 0;
    }
} // namespace ncnn
