//
// Created by GIBEREZ on 2025/12/31.
//

#ifndef NCNN_BATCHNORM_CUDA_H
#define NCNN_BATCHNORM_CUDA_H
#include "layer.h"

namespace ncnn {
    class BatchNorm_cuda : public Layer
    {
    public:
        BatchNorm_cuda();
        int load_param(const ParamDict& pd) override;
        int load_model(const ModelBin& mb) override;

        int forward_inplace(CudaMat& input_blob, const Option& opt) const override;

        int batchnorm_precompute();
        int batchnorm_cuda_inplace(CudaMat& input_blob) const;
    public:
        // param
        int channels;
        float eps;

        // model
        CudaMat slope_data;
        CudaMat mean_data;
        CudaMat var_data;
        CudaMat bias_data;

        CudaMat a_data;
        CudaMat b_data;
    };
}

#endif //NCNN_BATCHNORM_CUDA_H
