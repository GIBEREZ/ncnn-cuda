//
// Created by GIBEREZ on 2025/11/5.
//

#include "innerproduct_cuda.h"

namespace ncnn {

InnerProduct_cuda::InnerProduct_cuda()
{
    one_blob_only = true;
    support_cuda = true;
}

int InnerProduct_cuda::load_param(const ParamDict& pd)
{
    num_output = pd.get(0, 0);
    bias_term = pd.get(1, 0);
    weight_data_size = pd.get(2, 0);
    activation_type = pd.get(9, 0);
    activation_params = pd.get(10, Mat());
    return 0;
}

int InnerProduct_cuda::load_model(const ModelBin& mb)
{
    weight_blob = CudaMat(mb.load(weight_data_size, 0));
    if (weight_blob.empty())
    {
        NCNN_LOGE("===InnerProduct_cuda::load_model=== weight_blob.empty() failed");
        return -100;
    }

    if (bias_term)
    {
        bias_blob = CudaMat(mb.load(num_output, 0));
        if (bias_blob.empty())
        {
            NCNN_LOGE("===InnerProduct_cuda::load_model=== bias_blob.empty() failed");
            return -100;
        }
    }

    return 0;
}

int InnerProduct_cuda::upload_model(const Option& opt)
{
    return 0;
}

int InnerProduct_cuda::forward(const CudaMat& input_blob, CudaMat& output_blob, const Option& opt) const
{
    NCNN_LOGE("  *  Running CUDA InnerProduct forward  num_output=%d  weight_size=%d",
              num_output, weight_data_size);

    int ret = InnerProduct_cuda_forward(input_blob, output_blob, opt);
    if (ret != 0)
    {
        NCNN_LOGE("  *  CUDA InnerProduct forward FAILED ret=%d", ret);
        return ret;
    }

    NCNN_LOGE("  *  forward output_blob w=%d,h=%d,d=%d,c=%d,dims=%d",
              output_blob.w, output_blob.h, output_blob.d, output_blob.c, output_blob.dims);
    if (output_blob.empty() || output_blob.gpu_data == nullptr)
        NCNN_LOGE("  *  output blob gpu_data == nullptr");
    NCNN_LOGE("  *  CUDA InnerProduct forward done");
    return 0;
}

} // namespace ncnn
