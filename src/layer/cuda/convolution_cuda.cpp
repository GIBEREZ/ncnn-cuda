//
// Created by GIBEREZ on 2025/10/20.
//

#include "Convolution_cuda.h"

namespace ncnn {
    int Convolution_cuda::upload_model(const Option& opt)
    {
        Option option = opt;
        size_t weight_size = this->weight_data.total() * weight_data.elemsize;
        cudaMemcpy(weight_data.data_gpu, weight_data.data, weight_size, cudaMemcpyHostToDevice);

        if (bias_term)
        {
            size_t bias_size = bias_data.total() * bias_data.elemsize;
            cudaMemcpy(bias_data.data_gpu, bias_data.data, bias_size, cudaMemcpyHostToDevice);
        }
        return 0;
    }
    int Convolution_cuda::forward(const CudaMat& input_blob, CudaMat& output_blob, const Option& opt) const
    {
        Option option = opt;
        int c = input_blob.c;                   // 输入通道数
        int h = input_blob.h;                   // 输入高度
        int w = input_blob.w;                   // 输入宽度
        size_t elemsize = input_blob.elemsize;  // 精度

        if (input_blob.dims == 1 && kernel_w == 1 && kernel_h == 1)
        {
            return -1;
        }

        const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
        const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

        if (pad_left > 0 || pad_right > 0 || pad_top > 0 || pad_bottom > 0)
        {

        }

        return 0;
    }

    } // namespace ncnn