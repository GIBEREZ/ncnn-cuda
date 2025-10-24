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
        Layer* padding;         // ��Ե��0����
        CudaMat weight_data;    // ģ��Ȩ�ؾ�������
        CudaMat bias_data;      // ƫ�þ�������

        int num_output;         // ���ͨ����
        int kernel_w;           // ����˿��
        int kernel_h;           // ����˸߶�
        int pad_left;           // ��������Զ���ֵ
        int pad_right;          // �����ұ��Զ���ֵ
        int pad_top;            // ���������Զ���ֵ
        int pad_bottom;         // �����ױ��Զ���ֵ
        int dilation_w;         // ������ڿ�ȵ�����ϵ��
        int dilation_h;         // ������ڸ߶ȵ���ײϵ��
        int stride_w;           // ������ڿ�ȵĻ���������
        int stride_h;           // ������ڸ߶ȵĻ���������
        int weight_data_size;   // Ȩ�������ܴ�С��ͨ������ num_output * (input_channels/group) * kernel_h * kernel_w��
        int bias_term;          // �Ƿ�ʹ��ƫ���1��ʾ��bias��0��ʾû��
        int activation_type;    // ���������0 = �޼��1 = ReLU��2 = LeakyReLU��3 = Clip��4 = Sigmoid
    };
}

#endif //NCNN_CONV_CUDA_H
