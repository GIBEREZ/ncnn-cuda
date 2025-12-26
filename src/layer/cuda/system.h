#include "cuda_runtime.h"
#include <cuda_fp16.h>

namespace ncnn {
    // Forward declaration registration
    __global__ void Cast_kernel_FP32_to_FP16(float* input_blob, half* output_blob, int number);
    __global__ void Cast_kernel_FP16_to_FP32(half* input_blob, float* output_blob, int number);
    __global__ void Cast_kernel_FP32_to_INT8(float* input_blob, int8_t* output_blob, int number);
    __global__ void Cast_kernel_INT8_to_FP32(int8_t* input_blob, float* output_blob, int number);
    __global__ void Cast_kernel_FP16_to_INT8(half* input_blob, int8_t* output_blob, int number);
    __global__ void Cast_kernel_INT8_to_FP16(int8_t* input_blob, half* output_blob, int number);
}