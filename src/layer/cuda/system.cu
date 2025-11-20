#include "system.h"

namespace ncnn {
    __global__ void Cast_kernel_FP32_to_FP16(float* input_blob, half* output_blob, int number)
    {
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int idxElement = idx * 4;
        int elements_remaining = number - idxElement;
        if (elements_remaining >= 4) {
            output_blob[idxElement + 0] = __float2half(input_blob[idxElement + 0]);
            output_blob[idxElement + 1] = __float2half(input_blob[idxElement + 1]);
            output_blob[idxElement + 2] = __float2half(input_blob[idxElement + 2]);
            output_blob[idxElement + 3] = __float2half(input_blob[idxElement + 3]);
        } else {
            for (int i = 0; i < elements_remaining; i++) {
                output_blob[idxElement + i] = __float2half(input_blob[idxElement + i]);
            }
        }
    }

    __global__ void Cast_kernel_FP16_to_FP32(half* input_blob, float* output_blob, int number)
    {
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int idxElement = idx * 4;
        int elements_remaining = number - idxElement;
        if (elements_remaining >= 4) {
            output_blob[idxElement + 0] = __half2float(input_blob[idxElement + 0]);
            output_blob[idxElement + 1] = __half2float(input_blob[idxElement + 1]);
            output_blob[idxElement + 2] = __half2float(input_blob[idxElement + 2]);
            output_blob[idxElement + 3] = __half2float(input_blob[idxElement + 3]);
        } else {
            for (int i = 0; i < elements_remaining; i++) {
                output_blob[idxElement + i] = __half2float(input_blob[idxElement + i]);
            }
        }
    }
    __global__ void Cast_kernel_FP32_to_INT8(float* input_blob, half* output_blob, int number)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < number)
        {
            output_blob[idx] = __float2int_rz(input_blob[idx]);
        }
    }
    __global__ void Cast_kernel_INT8_to_FP32(int8_t* gpu_data);
    __global__ void Cast_kernel_FP16_to_INT8(half* gpu_data);
    __global__ void Cast_kernel_INT8_to_FP16(int8_t* gpu_data);
}