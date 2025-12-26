//
// Created by GIBEREZ on 2025/10/20.
//

#if NCNN_CUDA
    #include <iostream>
    #include <vector>
    #include "net.h"
    #include <cuda_runtime_api.h>
    #include <chrono>
    #include <command.h>

    int main() {
        ncnn::get_device_properties();

        ncnn::Net net;

        ncnn::Option option;
        option.use_cuda = true;

        ncnn::Net embed_net_;
        ncnn::Net encoder_net_;
        ncnn::Net decoder_net_;

        embed_net_.opt = option;
        encoder_net_.opt = option;
        decoder_net_.opt = option;

        int ret = encoder_net_.load_param("D:/software/Model/deepseek_r1_decoder.ncnn.param");
        if (ret != 0)
        {
            printf("Failed to load param file\n");
            return -1;
        }

        ret = encoder_net_.load_model("D:/software/Model/deepseek_r1_decoder.ncnn.bin.00");
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

        return 0;

        extractor.extract("out0", output);


        for (int i = 0; i < output.total(); i++)
        {
            NCNN_LOGE("%.6f ", output[i]);
        }

        return 0;
    }
#endif