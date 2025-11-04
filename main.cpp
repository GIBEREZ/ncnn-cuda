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

    ncnn::Net net;

    int ret = net.load_param("D:/software/Projects/C++Projects/ncnn-cuda/y8s-pig-detect-300e-3classes.param");
    if (ret != 0)
    {
        printf("Failed to load param file\n");
        return -1;
    }

    ncnn::Option option;
    option.use_cuda = true;

    ncnn::Extractor extractor = net.create_extractor();

    return 0;
}