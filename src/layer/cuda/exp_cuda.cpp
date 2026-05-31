//
// Exp CUDA layer implementation
//

#include "exp_cuda.h"

namespace ncnn {
    Exp_cuda::Exp_cuda()
    {
        one_blob_only = true;
        support_inplace = true;
        support_cuda = true;
    }

    int Exp_cuda::load_param(const ParamDict& pd)
    {
        base  = pd.get(0, -1.f);
        scale = pd.get(1, 1.f);
        shift = pd.get(2, 0.f);
        return 0;
    }

    int Exp_cuda::forward(const CudaMat& input_blob, CudaMat& output_blob, const Option& opt) const
    {
        NCNN_LOGE("  *  Running CUDA Exp forward  base=%.4f scale=%.4f shift=%.4f", base, scale, shift);
        exp_cuda(input_blob, output_blob, base, scale, shift);
        NCNN_LOGE("  *  forward output_blob w=%d,h=%d,d=%d,c=%d,dims=%d",output_blob.w,output_blob.h,output_blob.d,output_blob.c,output_blob.dims);
        NCNN_LOGE("  *  CUDA Exp forward done");
        return 0;
    }

    int Exp_cuda::forward_inplace(CudaMat& input_blob, const Option& opt) const
    {
        NCNN_LOGE("  *  Running CUDA Exp forward_inplace  base=%.4f scale=%.4f shift=%.4f", base, scale, shift);
        exp_cuda_inplace(input_blob, base, scale, shift);
        NCNN_LOGE("  *  forward output_blob w=%d,h=%d,d=%d,c=%d,dims=%d",input_blob.w,input_blob.h,input_blob.d,input_blob.c,input_blob.dims);
        NCNN_LOGE("  *  CUDA Exp forward_inplace done");
        return 0;
    }
} // namespace ncnn
