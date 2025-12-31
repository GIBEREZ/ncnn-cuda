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
        int forward(const CudaMat& input_blob, CudaMat& output_blob, const Option& opt) const override;
        int InnerProduct_cuda_forward(const CudaMat& input_blob, CudaMat& output_blob, const Option& opt) const;
    public:
        CudaMat weight_blob;    // Model weight matrix data
        CudaMat bias_blob;      // Bias matrix data

        int num_output;         // Number of output channels
        int bias_term;          // Whether to use a bias term; 1 = has bias, 0 = no bias
        int weight_data_size;   // Total size of the weight data, usually equal to num_output * (input_channels/group) * kernel_h * kernel_w
        int activation_type;    // Activation function type: 0 = none; 1 = ReLU; 2 = LeakyReLU; 3 = Clip; 4 = Sigmoid
        Mat activation_params;  // Parameters for the activation function
    };
}

#endif //NCNN_INNERPRODUCT_CUDA_H
