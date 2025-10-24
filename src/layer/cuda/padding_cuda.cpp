//
// Created by GIBEREZ on 2025/10/21.
//

#include "Padding_cuda.h"

namespace ncnn{
    Padding_cuda::Padding_cuda()
    {

    }

    int Padding_cuda::upload_model(const Option& opt)
    {
        return 0;
    }

    int Padding_cuda::forward(const CudaMat& input_blob, CudaMat& output_blob, const Option& opt) const
    {
        Option option = opt;
        int c = input_blob.c;                   // 输入通道数
        int h = input_blob.h;                   // 输入高度
        int w = input_blob.w;                   // 输入宽度
        size_t elemsize = input_blob.elemsize;  // 精度
        int elempack = input_blob.elempack;     // 打包数量

        int outw = 0;
        int outh = 0;

        if (input_blob.dims == 2)
        {
            //outw = w + pad_left + pad_right;
            //outh = h * elempack + pad_top + pad_bottom;
        }
        return 0;
    }

}
