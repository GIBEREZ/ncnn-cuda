//
// Created by GIBEREZ on 2025/10/20.
//

#include "Convolution_cuda.h"

namespace ncnn {
    Convolution_cuda::Convolution_cuda()
    {
        support_cuda = true;
        padding = 0;
    }

    int Convolution_cuda::load_param(const ParamDict& pd)
    {
        num_output = pd.get(0, 0);
        kernel_w = pd.get(1, 0);
        kernel_h = pd.get(11, kernel_w);
        pad_left = pd.get(4,0);
        pad_right = pd.get(15,pad_left);
        pad_top = pd.get(14, pad_left);
        pad_bottom = pd.get(16, pad_top);
        dilation_w = pd.get(2, 1);
        dilation_h = pd.get(12, dilation_w);
        stride_w = pd.get(3, 1);
        stride_h = pd.get(13, stride_w);
        dynamic_weight = pd.get(19, 0);
        weight_data_size = pd.get(6, 0);
        bias_term = pd.get(5, 0);
        activation_type = pd.get(9, 0);
        activation_params = pd.get(10, Mat());
        return 0;
    }

    int Convolution_cuda::load_model(const ModelBin& mb)
    {
        if (dynamic_weight)
        {
            return 0;
        }

        weight_data = CudaMat(mb.load(weight_data_size, 0));
        if (weight_data.empty())
            return -100;

        if (bias_term)
        {
            bias_data = CudaMat(mb.load(num_output, 0));
            if (bias_data.empty())
                return -100;
        }
        return 0;
    }

    int Convolution_cuda::upload_model(const Option& opt)
    {
        return 0;
    }

    int Convolution_cuda::forward(const CudaMat& input_blob, CudaMat& output_blob, const Option& opt) const
    {
        NCNN_LOGE("=== Running CUDA Convolution forward ==="); // µ˜ ‘”√
        Convolutio_cuda(input_blob, output_blob, opt);
        NCNN_LOGE("=== CUDA Convolution forward done ===");
        return 0;
    }

} // namespace ncnn