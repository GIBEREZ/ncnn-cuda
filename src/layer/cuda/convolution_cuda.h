//
// Created by GIBEREZ on 2025/10/20.
//

#ifndef NCNN_CONV_CUDA_H
#define NCNN_CONV_CUDA_H
// CUDA
#include "layer.h"

namespace ncnn {
    class Convolution_cuda : public Layer
    {
    public:
        Convolution_cuda();
        virtual int load_param(const ParamDict& pd);
        virtual int load_model(const ModelBin& mb);
        virtual int upload_model(const Option& opt);
        virtual int forward(const CudaMat& input_blob, CudaMat& output_blob, const Option& opt) const;
        int Convolution_cuda_forward(const CudaMat& input_blob, CudaMat& output_blob, const Option& opt) const;

    public:
        Layer* padding;
        CudaMat weight_blob;
        CudaMat bias_blob;

        int num_output;
        int kernel_w;
        int kernel_h;
        int pad_left;
        int pad_right;
        int pad_top;
        int pad_bottom;
        int dilation_w;
        int dilation_h;
        int stride_w;
        int stride_h;
        int dynamic_weight;
        int weight_data_size;
        int bias_term;
        int activation_type;
        Mat activation_params;
    };
}

#endif //NCNN_CONV_CUDA_H
