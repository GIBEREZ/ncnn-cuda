//
// Created by GIBEREZ on 2025/10/20.
//

#include <iostream>
#include <vector>
#include "net.h"
#include <cuda_runtime_api.h>
#include <command.h>

void print_mat(const ncnn::Mat& mat, int n = 10)
{
    if (mat.empty())
    {
        printf("Mat is empty\n");
        return;
    }

    int elemsize = mat.elemsize;
    int total_elements = mat.total();
    int channels = mat.c;
    int width = mat.w;
    int height = mat.h;

    printf("Mat: w=%d, h=%d, c=%d, total=%d, elemsize=%d\n", width, height, channels, total_elements, elemsize);

    int print_count = std::min(n, total_elements);

    if (elemsize == 1)
    {
        unsigned char* ptr = (unsigned char*)mat.data;
        for (int i = 0; i < print_count; i++)
            printf("mat[%d] = %u\n", i, ptr[i]);
    }
    else if (elemsize == 2)
    {
        uint16_t* ptr = (uint16_t*)mat.data;
        for (int i = 0; i < print_count; i++)
            printf("mat[%d] = %u\n", i, ptr[i]);
    }
    else if (elemsize == 4)
    {
        float* ptr = (float*)mat.data;
        for (int i = 0; i < print_count; i++)
            printf("mat[%d] = %f\n", i, ptr[i]);
    }
    else
    {
        printf("Unsupported element size: %d\n", elemsize);
    }
}

int main() {
    ncnn::get_device_properties();

    ncnn::Net net;

    ncnn::Option option;
    option.use_cuda = true;
    net.opt = option;

    int ret = net.load_param("D:/software/Projects/PythonProjects/torch/model.ncnn.param");
    if (ret != 0)
    {
        printf("Failed to load param file\n");
        return -1;
    }

    ret = net.load_model("D:/software/Projects/PythonProjects/torch/model.ncnn.bin");
    if (ret != 0)
    {
        printf("Failed to load model file\n");
        return -1;
    }

    const int width = 32;
    const int height = 32;
    const int channels = 3;

    std::srand((unsigned int)std::time(nullptr));

    std::vector<unsigned char> image_data(width * height * channels);
    for (auto& pixel : image_data)
    {
        pixel = static_cast<unsigned char>(std::rand() % 256);
    }

    ncnn::Mat A = ncnn::Mat::from_pixels_resize(
        image_data.data(),
        ncnn::Mat::PIXEL_RGB,
        width, height,
        32, 32
    );

    ncnn::CudaMat input(A);
    ncnn::Mat output;

    input.substract_mean_normalize(nullptr, nullptr);

    ncnn::Extractor extractor = net.create_extractor();

    extractor.input("in0", input);

    extractor.extract("out0", output);


    for (int i = 0; i < output.total(); i++)
    {
        NCNN_LOGE("%.6f ", output[i]);
    }

    return 0;
}