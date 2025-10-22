//
// Created by GIBEREZ on 2025/10/21.
//

#ifndef NCNN_PADDING_CUDA_H
#define NCNN_PADDING_CUDA_H
#include "layer.h"

namespace ncnn {
    class Padding_cuda : public Layer
    {
    public:
        Padding_cuda();
        virtual int upload_model(const Option& opt);
        virtual int forward(const CudaMat& input_blob, CudaMat& output_blob, const Option& opt) const;
    };

    int pad_left;           // ��������Զ���ֵ
    int pad_right;          // �����ұ��Զ���ֵ
    int pad_top;            // ���������Զ���ֵ
    int pad_bottom;         // �����ױ��Զ���ֵ
}


#endif //NCNN_PADDING_CUDA_H
