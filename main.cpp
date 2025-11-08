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
    // 打印设备信息（可选）
    ncnn::get_device_properties();

    // 创建网络对象
    ncnn::Net net;

    // 设置运行选项
    ncnn::Option option;
    option.use_cuda = true;    // 可启用 GPU 加速
    net.opt = option;

    // 加载参数文件
    int ret = net.load_param("D:/software/Projects/PythonProjects/torch/model.ncnn.param");
    if (ret != 0)
    {
        printf("Failed to load param file\n");
        return -1;
    }

    // 加载权重文件
    ret = net.load_model("D:/software/Projects/PythonProjects/torch/model.ncnn.bin");
    if (ret != 0)
    {
        printf("Failed to load model file\n");
        return -1;
    }

    // ================== 随机生成输入图片 ==================
    const int width = 32;
    const int height = 32;
    const int channels = 3;

    std::srand((unsigned int)std::time(nullptr));

    // 创建随机 RGB 图像数据 (范围 0~255)
    std::vector<unsigned char> image_data(width * height * channels);
    for (auto& pixel : image_data)
    {
        pixel = static_cast<unsigned char>(std::rand() % 256);
    }

    // 封装成 ncnn::Mat
    ncnn::Mat input = ncnn::Mat::from_pixels_resize(
        image_data.data(),        // 像素数据
        ncnn::Mat::PIXEL_RGB,     // 像素格式
        width, height,         // 原图尺寸
        32, 32        // 网络输入尺寸
    );

    // 可选：归一化（例如到 [0,1]）
    input.substract_mean_normalize(nullptr, nullptr);

    NCNN_LOGE("随机输入图像生成成功，尺寸: %d x %d x %d", width, height, channels);

    // ================== 网络推理 ==================
    ncnn::Extractor extractor = net.create_extractor();

    // 设置输入层（对应 param 文件中的名称）
    extractor.input("in0", input);
    NCNN_LOGE("输入层设置完毕");

    // 提取输出层
    ncnn::Mat output;
    extractor.extract("out0", output);

    // ================== 打印输出 ==================
    NCNN_LOGE("推理完成，输出尺寸: %d x %d x %d", output.c, output.h, output.w);
    for (int i = 0; i < output.total(); i++)
    {
        NCNN_LOGE("%.6f ", output[i]);
    }

    return 0;
}