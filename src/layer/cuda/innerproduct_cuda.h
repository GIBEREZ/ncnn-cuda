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
        virtual int load_param(const ParamDict& pd);
        virtual int load_model(const ModelBin& mb);
        virtual int forward(const CudaMat& input_blob, CudaMat& output_blob, const Option& opt) const;
        int InnerProduct_cuda_forward(const CudaMat& input_blob, CudaMat& output_blob, const Option& opt) const;
    public:
        CudaMat weight_data;    // 模型权重矩阵数据
        CudaMat bias_data;      // 偏置矩阵数据

        int num_output;         // 输出通道数
        int bias_term;          // 是否使用偏置项，1表示有bias，0表示没有
        int weight_data_size;   // 权重数据总大小，通常等于 num_output * (input_channels/group) * kernel_h * kernel_w。
        int activation_type;    // 激活函数类型0 = 无激活；1 = ReLU；2 = LeakyReLU；3 = Clip；4 = Sigmoid
        Mat activation_params;  // 激活函数保存的参数
    };
}

#endif //NCNN_INNERPRODUCT_CUDA_H
