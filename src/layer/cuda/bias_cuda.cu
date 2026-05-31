#include "bias_cuda.h"
#include <cuda_runtime.h>

namespace ncnn {

__global__ void bias_kernel_cuda(float* data, const float* bias, int channel_size, int channels, int cstep)
{
    // Each thread processes one element
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= channels * cstep) return;

    // Determine which channel this element belongs to
    int ch = idx / cstep;
    int offset_in_ch = idx - ch * cstep;

    // Only process valid elements (skip padding at end of each channel)
    if (offset_in_ch >= channel_size) return;

    data[idx] += bias[ch];
}

    int Bias_cuda::Bias_cuda_forward_inplace(CudaMat& input_blob) const
    {
        int C = 1, channel_size = 1;
        if (input_blob.dims == 1)      { C = 1;           channel_size = input_blob.w; }
        else if (input_blob.dims == 2) { C = input_blob.h; channel_size = input_blob.w; }
        else if (input_blob.dims == 3) { C = input_blob.c; channel_size = input_blob.h * input_blob.w; }
        else if (input_blob.dims == 4) { C = input_blob.c; channel_size = input_blob.d * input_blob.h * input_blob.w; }

        int cstep = (int)input_blob.cstep;
        int total = C * cstep;

        int threadsPerBlock = 1024;
        int blocksPerGrid = (total + threadsPerBlock - 1) / threadsPerBlock;

        bias_kernel_cuda<<<blocksPerGrid, threadsPerBlock>>>(
            static_cast<float*>(input_blob.gpu_data),
            static_cast<const float*>(bias_blob.gpu_data),
            channel_size, C, cstep);

        cudaDeviceSynchronize();
        return 0;
    }

}