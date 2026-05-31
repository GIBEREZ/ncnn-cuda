//
// HardSwish CUDA header
//

#ifndef NCNN_HARDSWISH_CUDA_H
#define NCNN_HARDSWISH_CUDA_H
#include "layer.h"

namespace ncnn {
int hardswish_cuda(const CudaMat& input_blob, CudaMat& output_blob, float alpha, float beta);
int hardswish_cuda_inplace(CudaMat& input_blob, float alpha, float beta);
    class HardSwish_cuda : public Layer
    {
    public:
        HardSwish_cuda();
        int load_param(const ParamDict& pd) override;
        using Layer::forward;
        int forward(const CudaMat& input_blob, CudaMat& output_blob, const Option& opt) const override;
        using Layer::forward_inplace;
        int forward_inplace(CudaMat& input_blob, const Option& opt) const override;
    public:
        float alpha;
        float beta;
    };
}

#endif //NCNN_HARDSWISH_CUDA_H
