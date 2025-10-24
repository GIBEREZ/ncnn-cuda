//
// Created by GIBEREZ on 2025/10/20.
//

#include <iostream>
#include <vector>
#include "net.h"
#include <cuda_runtime_api.h>
#include "layer/cuda/Relu_cuda.h"

int main() {
    // �������������� CudaMat
    ncnn::CudaMat input, output;
    input.create(640, 640, 3, sizeof(float));  // 640x640 ���룬3 ��ͨ��

    // ����һ�� ReLU ��
    ncnn::ReLU_cuda relu_layer;
    output.create(input.w, input.h, input.c, sizeof(float)); // ���� GPU �ڴ�
    // ǰ�򴫲�
    relu_layer.forward(input, output, ncnn::Option());
    std::vector<float> host_output(output.total());
    // ��ӡ�����һ����
    std::cout << "Output Tensor Info: " << output.w << " x " << output.h << " x " << output.c << std::endl;
    std::cout << "First 5 values in the output tensor: ";
    for (int i = 0; i < 5; i++) {
        std::cout << output.data_gpu[i] << " ";
    }
    std::cout << std::endl;

    // ������Դ
    input.release();
    output.release();

    return 0;
}