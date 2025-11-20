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

    __global__ void GEMM_kernel_float32(const void* input_blob, const void* weight_blob, void* output_blob, int M, int N, int K);
    __global__ void GEMM_kernel_half16(const void* input_blob, const void* weight_blob, void* output_blob, int M, int N, int K);
    __global__ void GEMM_kernel_int8(const void* input_blob, const void* weight_blob, void* output_blob, int M, int N, int K);

    struct LOAD {
        const void* gpu_data;
        int width;

        __host__ __device__ LOAD(const void* _gpu_data, int _width) : gpu_data(_gpu_data), width(_width) {}

        template<int PACK_SIZE>
        __host__ __device__ void load(float* vec, int row, int offset) const {
            const float* base = static_cast<const float*>(gpu_data);
            #pragma unroll
            for (int i = 0; i < PACK_SIZE; ++i) {
                vec[i] = base[row * width + offset + i];
            }
        }

        template<int PACK_SIZE>
        __host__ __device__ void load(__half* vec, int row, int offset) const {
            const __half* base = static_cast<const __half*>(gpu_data);
            #pragma unroll
            for (int i = 0; i < PACK_SIZE; ++i) {
                vec[i] = base[row * width + offset + i];
            }
        }

        template<int PACK_SIZE>
        __host__ __device__ void load(int8_t* vec, int row, int offset) const {
            const int8_t* base = static_cast<const int8_t*>(gpu_data);
            #pragma unroll
            for (int i = 0; i < PACK_SIZE; ++i) {
                vec[i] = base[row * width + offset + i];
            }
        }
    };

    struct STORE {
        void* gpu_data;
        int width;

        __host__ __device__ STORE(void* _gpu_data, int _width) : gpu_data(_gpu_data), width(_width) {}

        template<int PACK_SIZE>
        __host__ __device__ void store(const float* vec, int row, int offset) const {
            float* base = static_cast<float*>(gpu_data);
            #pragma unroll
            for (int i = 0; i < PACK_SIZE; ++i) {
                base[row * width + offset + i] = vec[i];
            }
        }

        template<int PACK_SIZE>
        __host__ __device__ void store(const __half* vec, int row, int offset) const {
            __half* base = static_cast<__half*>(gpu_data);
            #pragma unroll
            for (int i = 0; i < PACK_SIZE; ++i) {
                base[row * width + offset + i] = vec[i];
            }
        }

        template<int PACK_SIZE>
        __host__ __device__ void store(const int8_t* vec, int row, int offset) const {
            int8_t* base = static_cast<int8_t*>(gpu_data);
            #pragma unroll
            for (int i = 0; i < PACK_SIZE; ++i) {
                base[row * width + offset + i] = vec[i];
            }
        }
    };
}