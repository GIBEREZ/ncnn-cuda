//
// Mish CUDA header
//

#ifndef NCNN_MISH_CUDA_H
#define NCNN_MISH_CUDA_H
#include "layer.h"

namespace ncnn {
int mish_cuda(const CudaMat& input_blob, CudaMat& output_blob);
int mish_cuda_inplace(CudaMat& input_blob);
    class Mish_cuda : public Layer
    {
    public:
        Mish_cuda();
        using Layer::forward;
        int forward(const CudaMat& input_blob, CudaMat& output_blob, const Option& opt) const override;
        using Layer::forward_inplace;
        int forward_inplace(CudaMat& input_blob, const Option& opt) const override;
    };
}

#endif //NCNN_MISH_CUDA_H
