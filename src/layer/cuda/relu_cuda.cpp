//
// Created by GIBEREZ on 2025/10/22.
//

#include "Relu_cuda.h"

#include <iostream>
#include <ostream>

namespace ncnn {
    ReLU_cuda::ReLU_cuda()
    {
        // �Ƿ�ò�ֻ����������blob��������
        one_blob_only = true;
        // �Ƿ�֧��ԭ�ؼ���
        support_inplace = true;
    }
    int ReLU_cuda::load_param(const ParamDict& pd)
    {
        slope = pd.get(0, 0.f);
        return 0;
    }

    int ReLU_cuda::upload_model(const Option& opt)
    {
        return 0;
    }

    int ReLU_cuda::forward(const CudaMat& input_blob, CudaMat& output_blob, const Option& opt) const
    {
        relu_cuda(input_blob, output_blob, input_blob.total());
        return 0;
    }
} // namespace ncnn
