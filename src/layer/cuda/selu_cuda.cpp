//
// SELU CUDA layer implementation
//

#include "selu_cuda.h"

namespace ncnn {
    SELU_cuda::SELU_cuda()
    {
        one_blob_only = true;
        support_inplace = true;
        support_cuda = true;
    }

    int SELU_cuda::load_param(const ParamDict& pd)
    {
        alpha  = pd.get(0, 1.673264f);
        lambda = pd.get(1, 1.050700f);
        return 0;
    }

    int SELU_cuda::forward(const CudaMat& input_blob, CudaMat& output_blob, const Option& opt) const
    {
        NCNN_LOGE("  *  Running CUDA SELU forward  alpha=%.4f lambda=%.4f", alpha, lambda);
        selu_cuda(input_blob, output_blob, alpha, lambda);
        NCNN_LOGE("  *  forward output_blob w=%d,h=%d,d=%d,c=%d,dims=%d",output_blob.w,output_blob.h,output_blob.d,output_blob.c,output_blob.dims);
        NCNN_LOGE("  *  CUDA SELU forward done");
        return 0;
    }

    int SELU_cuda::forward_inplace(CudaMat& input_blob, const Option& opt) const
    {
        NCNN_LOGE("  *  Running CUDA SELU forward_inplace  alpha=%.4f lambda=%.4f", alpha, lambda);
        selu_cuda_inplace(input_blob, alpha, lambda);
        NCNN_LOGE("  *  forward output_blob w=%d,h=%d,d=%d,c=%d,dims=%d",input_blob.w,input_blob.h,input_blob.d,input_blob.c,input_blob.dims);
        NCNN_LOGE("  *  CUDA SELU forward_inplace done");
        return 0;
    }
} // namespace ncnn
