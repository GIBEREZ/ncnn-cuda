//
// Mish CUDA layer implementation
//

#include "mish_cuda.h"

namespace ncnn {
    Mish_cuda::Mish_cuda()
    {
        one_blob_only = true;
        support_inplace = true;
        support_cuda = true;
    }

    int Mish_cuda::forward(const CudaMat& input_blob, CudaMat& output_blob, const Option& opt) const
    {
        NCNN_LOGE("  *  Running CUDA Mish forward");
        mish_cuda(input_blob, output_blob);
        NCNN_LOGE("  *  forward output_blob w=%d,h=%d,d=%d,c=%d,dims=%d",output_blob.w,output_blob.h,output_blob.d,output_blob.c,output_blob.dims);
        NCNN_LOGE("  *  CUDA Mish forward done");
        return 0;
    }

    int Mish_cuda::forward_inplace(CudaMat& input_blob, const Option& opt) const
    {
        NCNN_LOGE("  *  Running CUDA Mish forward_inplace");
        mish_cuda_inplace(input_blob);
        NCNN_LOGE("  *  forward output_blob w=%d,h=%d,d=%d,c=%d,dims=%d",input_blob.w,input_blob.h,input_blob.d,input_blob.c,input_blob.dims);
        NCNN_LOGE("  *  CUDA Mish forward_inplace done");
        return 0;
    }
} // namespace ncnn
