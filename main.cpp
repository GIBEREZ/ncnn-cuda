//
// Created by GIBEREZ on 2025/10/20.
//

#include <iostream>
#include <vector>
#include "net.h"
#include <cuda_runtime_api.h>
#include <command.h>
#include "layer/cuda/Relu_cuda.h"

int main() {
    ncnn::get_device_properties();

    // �������������� CudaMat
    ncnn::CudaMat input, output;
    input.create(640, 640, 3, sizeof(float));  // 640x640 ���룬3 ��ͨ��

    // ����һ�� ReLU ��
    ncnn::ReLU_cuda relu_layer;
    output.create(input.w, input.h, input.c, sizeof(float)); // ���� GPU �ڴ�
    // ǰ�򴫲�
    relu_layer.forward(input, output, ncnn::Option());
    output.download(output);
    std::vector<float> host_output(output.total());

    // ������Դ
    input.release();
    output.release();

    return 0;
}