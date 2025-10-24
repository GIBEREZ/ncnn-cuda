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
        virtual int upload_model(const Option& opt);
        virtual int forward(const CudaMat& input_blob, CudaMat& output_blob, const Option& opt) const;
    public:
        Layer* padding;         // 边缘补0操作
        CudaMat weight_data;    // 模型权重矩阵数据
        CudaMat bias_data;      // 偏置矩阵数据

        int num_output;         // 输出通道数
        int kernel_w;           // 卷积核宽度
        int kernel_h;           // 卷积核高度
        int pad_left;           // 张量左边自动补值
        int pad_right;          // 张量右边自动补值
        int pad_top;            // 张量顶边自动补值
        int pad_bottom;         // 张量底边自动补值
        int dilation_w;         // 卷积核在宽度的膨胀系数
        int dilation_h;         // 卷积核在高度的碰撞系数
        int stride_w;           // 卷积核在宽度的滑动步长。
        int stride_h;           // 卷积核在高度的滑动步长。
        int weight_data_size;   // 权重数据总大小，通常等于 num_output * (input_channels/group) * kernel_h * kernel_w。
        int bias_term;          // 是否使用偏置项，1表示有bias，0表示没有
        int activation_type;    // 激活函数类型0 = 无激活；1 = ReLU；2 = LeakyReLU；3 = Clip；4 = Sigmoid
    };
}

#endif //NCNN_CONV_CUDA_H
