//
// Created by GIBEREZ on 2025/10/21.
//

#include "Padding_cuda.h"

namespace ncnn{
    Padding_cuda::Padding_cuda()
    {
        one_blob_only = true;
        support_cuda = true;
    }

    int Padding_cuda::upload_model(const Option& opt)
    {
        return 0;
    }

    int Padding_cuda::forward(const CudaMat& input_blob, CudaMat& output_blob, const Option& opt) const
    {
        Option option = opt;
        int c = input_blob.c;                   // input channels
        int h = input_blob.h;                   // input height
        int w = input_blob.w;                   // input width
        size_t elemsize = input_blob.elemsize;  // element size
        int elempack = input_blob.elempack;     // element pack count

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
