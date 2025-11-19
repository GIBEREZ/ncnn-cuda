//
// Created by GIBEREZ on 2025/10/20.
//

#include <iostream>
#include <vector>
#include "net.h"
#include "layer/cuda/relu_cuda.h"
#include "layer/cuda/softmax_cuda.h"

#include <cuda_runtime_api.h>
#include <chrono>
#include <command.h>

void Layer_GFLOPS()
{
    int width = 512;
    int height = 512;
    int channels = 512;

    int loop_count = 2000;

    ncnn::Option option;
    std::vector<float> input_data(width * height * channels);

    for (auto& v : input_data)
        v = rand() / float(RAND_MAX);

    ncnn::CudaMat input_blob(width, height, channels, input_data.data(), 4);
    ncnn::CudaMat output_blob(width, height, channels, 4);

    ncnn::ReLU_cuda layer;

    ncnn::cudaEvent_t start_event, end_event;
    ncnn::cudaEventCreate(&start_event);
    ncnn::cudaEventCreate(&end_event);

    ncnn::cudaEventRecord(start_event);

    for (int i = 0; i < loop_count; i++)
    {
        layer.forward(input_blob, output_blob, option);
    }

    ncnn::cudaEventRecord(end_event);
    ncnn::cudaEventSynchronize(end_event);

    float elapsed_ms = 0.f;
    ncnn::cudaEventElapsedTime(&elapsed_ms, start_event, end_event);

    int N = width * height * channels;

    double flops_per_softmax = 6.0 * N;

    double total_flops = flops_per_softmax * loop_count;

    double elapsed_sec = elapsed_ms / 1000.0;

    double gflops = total_flops / elapsed_sec / 1e9;

    NCNN_LOGE("Softmax Size = %d x %d x %d", width, height, channels);
    NCNN_LOGE("loop = %d", loop_count);
    NCNN_LOGE("Elapsed = %.3f ms", elapsed_ms);
    NCNN_LOGE("GFLOPS = %.3f", gflops);

    ncnn::cudaEventDestroy(start_event);
    ncnn::cudaEventDestroy(end_event);
}

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

    Layer_GFLOPS();

    return 0;

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