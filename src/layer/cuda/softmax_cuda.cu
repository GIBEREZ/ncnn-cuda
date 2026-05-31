// Softmax CUDA kernel
// Each thread handles one group (e.g., one spatial position for channel-wise softmax).
// Uses pre-computed offsets to handle ncnn's cstep-aligned tensor layout.
// Created by GIBEREZ on 2025/12/26.

#include "softmax_cuda.h"
#include <float.h>
#include <vector>

namespace ncnn {

__global__ void softmax_kernel_cuda(float* input, float* output, int* offsets,
                                     int axis_size, int stride, int num_groups)
{
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= num_groups) return;

    int base = offsets[g];

    // Step 1: find max (numerical stability)
    float max_val = -FLT_MAX;
    for (int a = 0; a < axis_size; a++)
    {
        float val = input[base + a * stride];
        if (val > max_val) max_val = val;
    }

    // Step 2: compute exp and sum
    float sum = 0.0f;
    for (int a = 0; a < axis_size; a++)
    {
        float val = expf(input[base + a * stride] - max_val);
        output[base + a * stride] = val;
        sum += val;
    }

    // Step 3: normalize
    float inv_sum = 1.0f / sum;
    for (int a = 0; a < axis_size; a++)
        output[base + a * stride] *= inv_sum;
}

// Compute the base offset for each softmax group
static void compute_offsets(int dims, int w, int h, int d, int c, int cstep,
                             int axis, int num_groups, std::vector<int>& offsets)
{
    offsets.resize(num_groups);
    int g = 0;

    if (dims == 1)
    {
        offsets[0] = 0;
    }
    else if (dims == 2)
    {
        if (axis == 0)
            for (int iw = 0; iw < w; iw++) offsets[g++] = iw;
        else
            for (int ih = 0; ih < h; ih++) offsets[g++] = ih * w;
    }
    else if (dims == 3)
    {
        if (axis == 0)
            for (int ih = 0; ih < h; ih++)
                for (int iw = 0; iw < w; iw++)
                    offsets[g++] = ih * w + iw;
        else if (axis == 1)
            for (int ic = 0; ic < c; ic++)
                for (int iw = 0; iw < w; iw++)
                    offsets[g++] = ic * cstep + iw;
        else
            for (int ic = 0; ic < c; ic++)
                for (int ih = 0; ih < h; ih++)
                    offsets[g++] = ic * cstep + ih * w;
    }
    else // dims == 4
    {
        if (axis == 0)
            for (int id = 0; id < d; id++)
                for (int ih = 0; ih < h; ih++)
                    for (int iw = 0; iw < w; iw++)
                        offsets[g++] = id * h * w + ih * w + iw;
        else if (axis == 1)
            for (int ic = 0; ic < c; ic++)
                for (int ih = 0; ih < h; ih++)
                    for (int iw = 0; iw < w; iw++)
                        offsets[g++] = ic * cstep + ih * w + iw;
        else if (axis == 2)
            for (int ic = 0; ic < c; ic++)
                for (int id = 0; id < d; id++)
                    for (int iw = 0; iw < w; iw++)
                        offsets[g++] = ic * cstep + id * h * w + iw;
        else
            for (int ic = 0; ic < c; ic++)
                for (int id = 0; id < d; id++)
                    for (int ih = 0; ih < h; ih++)
                        offsets[g++] = ic * cstep + id * h * w + ih * w;
    }
}

// Helper: determine softmax parameters (axis_size, num_groups, stride)
static void softmax_params(int dims, int w, int h, int d, int c, int cstep, int axis,
                            int& axis_size, int& num_groups, int& stride)
{
    if (dims == 1)
    {
        axis_size = w; num_groups = 1; stride = 1;
    }
    else if (dims == 2)
    {
        if (axis == 0)      { axis_size = h; num_groups = w;  stride = w; }
        else /* axis == 1 */ { axis_size = w; num_groups = h;  stride = 1; }
    }
    else if (dims == 3)
    {
        if (axis == 0)      { axis_size = c; num_groups = h * w; stride = cstep; }
        else if (axis == 1) { axis_size = h; num_groups = c * w; stride = w; }
        else                { axis_size = w; num_groups = c * h; stride = 1; }
    }
    else // dims == 4
    {
        if (axis == 0)      { axis_size = c; num_groups = d * h * w; stride = cstep; }
        else if (axis == 1) { axis_size = d; num_groups = c * h * w; stride = h * w; }
        else if (axis == 2) { axis_size = h; num_groups = c * d * w; stride = w; }
        else                { axis_size = w; num_groups = c * d * h; stride = 1; }
    }
}

int softmax_cuda(const CudaMat& input_blob, CudaMat& output_blob, int axis)
{
    int dims = input_blob.dims;
    int w = input_blob.w, h = input_blob.h, d = input_blob.d, c = input_blob.c;
    int cstep = (int)input_blob.cstep;

    if (axis < 0) axis += dims;

    int axis_size, num_groups, stride;
    softmax_params(dims, w, h, d, c, cstep, axis, axis_size, num_groups, stride);

    if (output_blob.empty())
        output_blob.create_like(input_blob);

    std::vector<int> offsets;
    compute_offsets(dims, w, h, d, c, cstep, axis, num_groups, offsets);

    int* d_offsets = nullptr;
    cudaMalloc(&d_offsets, num_groups * sizeof(int));
    cudaMemcpy(d_offsets, offsets.data(), num_groups * sizeof(int), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (num_groups + threads - 1) / threads;
    softmax_kernel_cuda<<<blocks, threads>>>(
        static_cast<float*>(const_cast<void*>(input_blob.gpu_data)),
        static_cast<float*>(output_blob.gpu_data),
        d_offsets, axis_size, stride, num_groups);

    cudaDeviceSynchronize();
    cudaFree(d_offsets);
    return 0;
}

int softmax_cuda_inplace(CudaMat& input_blob, int axis)
{
    int dims = input_blob.dims;
    int w = input_blob.w, h = input_blob.h, d = input_blob.d, c = input_blob.c;
    int cstep = (int)input_blob.cstep;

    if (axis < 0) axis += dims;

    int axis_size, num_groups, stride;
    softmax_params(dims, w, h, d, c, cstep, axis, axis_size, num_groups, stride);

    std::vector<int> offsets;
    compute_offsets(dims, w, h, d, c, cstep, axis, num_groups, offsets);

    int* d_offsets = nullptr;
    cudaMalloc(&d_offsets, num_groups * sizeof(int));
    cudaMemcpy(d_offsets, offsets.data(), num_groups * sizeof(int), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (num_groups + threads - 1) / threads;
    float* gpu_data = static_cast<float*>(input_blob.gpu_data);
    softmax_kernel_cuda<<<blocks, threads>>>(
        gpu_data, gpu_data,
        d_offsets, axis_size, stride, num_groups);

    cudaDeviceSynchronize();
    cudaFree(d_offsets);
    return 0;
}

} // namespace ncnn