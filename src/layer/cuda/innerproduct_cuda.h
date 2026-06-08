//
// Created by GIBEREZ on 2025/11/5.
//

#ifndef NCNN_INNERPRODUCT_CUDA_H
#define NCNN_INNERPRODUCT_CUDA_H

#include "layer.h"

namespace ncnn
{

class InnerProduct_cuda : public Layer
{
public:
    InnerProduct_cuda();
    int load_param(const ParamDict& pd) override;
    int load_model(const ModelBin& mb) override;
    int upload_model(const Option& opt) override;
    using Layer::forward;
    int forward(const CudaMat& input_blob, CudaMat& output_blob, const Option& opt) const override;
    int InnerProduct_cuda_forward(const CudaMat& input_blob, CudaMat& output_blob, const Option& opt) const;

public:
    CudaMat weight_blob;
    CudaMat bias_blob;

    int num_output;
    int bias_term;
    int weight_data_size;
    int activation_type;
    Mat activation_params;
};

} // namespace ncnn

#endif //NCNN_INNERPRODUCT_CUDA_H
