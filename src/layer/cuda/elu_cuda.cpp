//
// ELU CUDA layer implementation
//

#include "elu_cuda.h"

namespace ncnn {
    ELU_cuda::ELU_cuda()
    {
        one_blob_only = true;
        support_inplace = true;
        support_cuda = true;
    }

    int ELU_cuda::load_param(const ParamDict& pd)
    {
        alpha = pd.get(0, 0.1f);
        return 0;
    }

    int ELU_cuda::forward(const CudaMat& input_blob, CudaMat& output_blob, const Option& opt) const
    {
        NCNN_LOGE("  *  Running CUDA ELU forward  alpha=%.4f", alpha);
        elu_cuda(input_blob, output_blob, alpha);
        NCNN_LOGE("  *  forward output_blob w=%d,h=%d,d=%d,c=%d,dims=%d",output_blob.w,output_blob.h,output_blob.d,output_blob.c,output_blob.dims);
        NCNN_LOGE("  *  CUDA ELU forward done");
        return 0;
    }

    int ELU_cuda::forward_inplace(CudaMat& input_blob, const Option& opt) const
    {
        NCNN_LOGE("  *  Running CUDA ELU forward_inplace  alpha=%.4f", alpha);
        elu_cuda_inplace(input_blob, alpha);
        NCNN_LOGE("  *  forward output_blob w=%d,h=%d,d=%d,c=%d,dims=%d",input_blob.w,input_blob.h,input_blob.d,input_blob.c,input_blob.dims);
        NCNN_LOGE("  *  CUDA ELU forward_inplace done");
        return 0;
    }
} // namespace ncnn
