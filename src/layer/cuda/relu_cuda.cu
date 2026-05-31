#include "relu_cuda.h"
#include <cuda_runtime.h>

namespace ncnn {
    /**
     * CUDA ReLU kernel and its C++ API interface.
     * @param input_blob input tensor - linear array pointer
     * @param output_blob output tensor - linear array pointer
     * @param number number of elements in the linear array
     * In CUDA kernel, x (const float* x) is just a pointer to a contiguous memory
     * linear array; it neither knows nor cares about dimensions (shape).
     */
    __global__ void relu_kernel_cuda(const float* input_blob, float* output_blob, int number, float slope)
    {
        // calculate global thread index
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int idxElement = idx * 4;
        if (idxElement + 3 < number) {
            float4 vec_in = *(float4*)&input_blob[idxElement];
            float4 vec_out;

            vec_out.x = vec_in.x > 0.0f ? vec_in.x : vec_in.x * slope;
            vec_out.y = vec_in.y > 0.0f ? vec_in.y : vec_in.y * slope;
            vec_out.z = vec_in.z > 0.0f ? vec_in.z : vec_in.z * slope;
            vec_out.w = vec_in.w > 0.0f ? vec_in.w : vec_in.w * slope;

            *(float4*)&output_blob[idxElement] = vec_out;
        }
        else {
            for (int i = 0; i < 4; i++) {
                unsigned int elem_idx = idxElement + i;
                if (elem_idx < number) {
                    float val = input_blob[elem_idx];
                    output_blob[elem_idx] = val > 0.0f ? val : val * slope;
                }
            }
        }
    }
    void relu_cuda(const CudaMat& input_blob, CudaMat& output_blob, int number, float slope)
    {
        int threadsPerBlock = 1024;                                                         // threads per block
        int vec_size = 16 / input_blob.elemsize;                                            // elements per thread (guarantee 16B alignment)
        int totalThreadsNeeded = (number + vec_size - 1) / vec_size;                        // total threads needed for the entire array
        int blocksPerGrid = (totalThreadsNeeded + threadsPerBlock - 1) / threadsPerBlock;   // number of thread blocks in grid (ceil division)

        relu_kernel_cuda<<<blocksPerGrid, threadsPerBlock>>>(static_cast<const float*>(input_blob.gpu_data), static_cast<float*>(output_blob.gpu_data), number, slope);
        // synchronize device; wait for kernel completion. blocks host (CPU) until all device (GPU) operations finish.
        cudaDeviceSynchronize();
    }
}