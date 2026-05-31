//
// GELU CUDA layer implementation
//

#include "gelu_cuda.h"

namespace ncnn {
    GELU_cuda::GELU_cuda()
    {
        one_blob_only = true;
        support_inplace = true;
        support_cuda = true;
    }

    int GELU_cuda::load_param(const ParamDict& pd)
    {
        fast_gelu = pd.get(0, 0) != 0;
        return 0;
    }

    int GELU_cuda::forward(const CudaMat& input_blob, CudaMat& output_blob, const Option& opt) const
    {
        NCNN_LOGE("  *  Running CUDA GELU forward  fast_gelu=%d", (int)fast_gelu);
        gelu_cuda(input_blob, output_blob, fast_gelu);
        NCNN_LOGE("  *  forward output_blob w=%d,h=%d,d=%d,c=%d,dims=%d",output_blob.w,output_blob.h,output_blob.d,output_blob.c,output_blob.dims);
        NCNN_LOGE("  *  CUDA GELU forward done");
        return 0;
    }

    int GELU_cuda::forward_inplace(CudaMat& input_blob, const Option& opt) const
    {
        NCNN_LOGE("  *  Running CUDA GELU forward_inplace  fast_gelu=%d", (int)fast_gelu);
        gelu_cuda_inplace(input_blob, fast_gelu);
        NCNN_LOGE("  *  forward output_blob w=%d,h=%d,d=%d,c=%d,dims=%d",input_blob.w,input_blob.h,input_blob.d,input_blob.c,input_blob.dims);
        NCNN_LOGE("  *  CUDA GELU forward_inplace done");
        return 0;
    }
} // namespace ncnn
