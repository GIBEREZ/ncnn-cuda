//
// Created by GIBEREZ on 2025/11/4.
// CUDA custom convolution implementation (im2col + tiled GEMM)
//
#include "convolution_cuda.h"
#include "system.h"
#include <cuda_runtime.h>
#include <cfloat>

namespace ncnn {

// ============================================================================
// Kernel 1: im2col – flatten each convolution patch into a column
// Input:  [IC, IH, IW]  with channel stride in_cstep
// Output: col [K, N] row-major, where K = IC*KH*KW, N = OH*OW
//         col[k * N + n] corresponds to kernel element k at output position n
// ============================================================================
__global__ void im2col_kernel(
    const float* input,
    float* col,
    int inch,
    int inh,
    int inw,
    int in_cstep,
    int outh,
    int outw,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int dilation_h,
    int dilation_w,
    int pad_top,
    int pad_left,
    int K,            // K = inch * kernel_h * kernel_w
    int N)            // N = outh * outw
{
    // Each thread handles one element of the im2col matrix
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = K * N;
    if (idx >= total)
        return;

    // Decode kernel element index k and output position n
    int k_idx = idx / N;
    int n_idx = idx % N;

    int ic = k_idx / (kernel_h * kernel_w);
    int k_rem = k_idx % (kernel_h * kernel_w);
    int ky = k_rem / kernel_w;
    int kx = k_rem % kernel_w;

    int oh = n_idx / outw;
    int ow = n_idx % outw;

    // Compute input position with stride, dilation, and padding
    int ih = oh * stride_h + ky * dilation_h - pad_top;
    int iw = ow * stride_w + kx * dilation_w - pad_left;

    if (ih >= 0 && ih < inh && iw >= 0 && iw < inw)
        col[idx] = input[ic * in_cstep + ih * inw + iw];
    else
        col[idx] = 0.0f;
}

// ============================================================================
// Kernel 2: Tiled GEMM  C[M][N] = alpha * A[M][K] * B[K][N] + beta * C[M][N]
// Uses shared-memory tiling for performance.
// Tile size: TILE_M x TILE_N, with K dimension tiled by TILE_K.
// ============================================================================
#define TILE_M 16
#define TILE_N 16
#define TILE_K 16

__global__ void gemm_kernel_nn(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K,
    float alpha,
    float beta)
{
    // Block indices determine which tile of C this block computes
    int block_row = blockIdx.y;
    int block_col = blockIdx.x;

    // Thread indices within the block
    int thread_row = threadIdx.y;
    int thread_col = threadIdx.x;

    // Shared memory for tiles of A and B
    __shared__ float As[TILE_M][TILE_K];
    __shared__ float Bs[TILE_K][TILE_N];

    // Accumulator for this thread's C element
    float sum = 0.0f;

    // Loop over K dimension in tiles
    for (int k_block = 0; k_block < (K + TILE_K - 1) / TILE_K; k_block++)
    {
        // Load A tile into shared memory
        int a_row = block_row * TILE_M + thread_row;
        int a_col = k_block * TILE_K + thread_col;
        if (a_row < M && a_col < K)
            As[thread_row][thread_col] = A[a_row * K + a_col];
        else
            As[thread_row][thread_col] = 0.0f;

        // Load B tile into shared memory
        int b_row = k_block * TILE_K + thread_row;
        int b_col = block_col * TILE_N + thread_col;
        if (b_row < K && b_col < N)
            Bs[thread_row][thread_col] = B[b_row * N + b_col];
        else
            Bs[thread_row][thread_col] = 0.0f;

        __syncthreads();

        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < TILE_K; k++)
        {
            sum += As[thread_row][k] * Bs[k][thread_col];
        }

        __syncthreads();
    }

    // Write result to C
    int c_row = block_row * TILE_M + thread_row;
    int c_col = block_col * TILE_N + thread_col;
    if (c_row < M && c_col < N)
    {
        if (beta == 0.0f)
            C[c_row * N + c_col] = alpha * sum;
        else
            C[c_row * N + c_col] = alpha * sum + beta * C[c_row * N + c_col];
    }
}

// ============================================================================
// Kernel 3: Bias + Activation (fused)
// Applies bias and activation to the output tensor in-place.
// Layout: output[OC, OH, OW] with channel stride out_cstep
// ============================================================================
__global__ void bias_activation_kernel(
    float* output,
    const float* bias,
    int outch,
    int outh,
    int outw,
    int out_cstep,
    int bias_term,
    int activation_type,
    float act_param0,
    float act_param1)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outch * outh * outw;
    if (idx >= total)
        return;

    int oc = idx / (outh * outw);
    int spatial = idx % (outh * outw);
    int oh = spatial / outw;
    int ow = spatial % outw;

    float val = output[oc * out_cstep + oh * outw + ow];

    // Apply bias
    if (bias_term)
        val += bias[oc];

    // Apply activation
    switch (activation_type)
    {
    case 0: // None
        break;
    case 1: // ReLU
        val = val > 0.0f ? val : 0.0f;
        break;
    case 2: // LeakyReLU
        val = val > 0.0f ? val : val * act_param0;
        break;
    case 3: // Clip
        val = val < act_param0 ? act_param0 : val;
        val = val > act_param1 ? act_param1 : val;
        break;
    case 4: // Sigmoid
        val = 1.0f / (1.0f + expf(-val));
        break;
    case 5: // Mish
        val = val * tanhf(logf(expf(val) + 1.0f));
        break;
    case 6: // HardSwish
        {
            float alpha = act_param0;
            float beta = act_param1;
            float lower = -beta / alpha;
            float upper = (1.0f / alpha) + lower;
            if (val < lower)
                val = 0.0f;
            else if (val > upper)
                ;
            else
                val = val * (val * alpha + beta);
        }
        break;
    default:
        break;
    }

    output[oc * out_cstep + oh * outw + ow] = val;
}

// ============================================================================
// Host-side helper: im2col wrapper (raw pointer version)
// Input:  float* pointing to [IC, IH, IW] with cstep channel stride
// Output: CudaMat col_blob [K, N] row-major = B[K][N]
// ============================================================================
static void convolution_cuda_im2col_raw(
    const float* input_gpu,
    CudaMat& col_blob,
    int inch,
    int inh,
    int inw,
    int in_cstep,
    int kernel_w,
    int kernel_h,
    int stride_w,
    int stride_h,
    int dilation_w,
    int dilation_h,
    int pad_left,
    int pad_top,
    int outw,
    int outh)
{
    int K = inch * kernel_h * kernel_w;
    int N = outh * outw;

    col_blob.release();
    col_blob.create(N, K, sizeof(float));   // w=N, h=K  →  B[K][N] row-major
    if (col_blob.empty())
        return;

    int total = K * N;
    int threads_per_block = 256;
    int blocks_per_grid = (total + threads_per_block - 1) / threads_per_block;

    im2col_kernel<<<blocks_per_grid, threads_per_block>>>(
        input_gpu,
        static_cast<float*>(col_blob.gpu_data),
        inch, inh, inw, in_cstep,
        outh, outw,
        kernel_h, kernel_w,
        stride_h, stride_w,
        dilation_h, dilation_w,
        pad_top, pad_left,
        K, N);

    cudaDeviceSynchronize();
}

// ============================================================================
// Host-side helper: GEMM wrapper
// C = alpha * A * B + beta * C
// A: M x K    B: K x N    C: M x N
// ============================================================================
static void convolution_cuda_gemm(
    const CudaMat& A,
    const CudaMat& B,
    CudaMat& C,
    int M,
    int N,
    int K,
    int /*transA*/,
    int /*transB*/)
{
    float alpha = 1.0f;
    float beta = 0.0f;

    dim3 block_dim(TILE_N, TILE_M);
    dim3 grid_dim(
        (N + TILE_N - 1) / TILE_N,
        (M + TILE_M - 1) / TILE_M);

    gemm_kernel_nn<<<grid_dim, block_dim>>>(
        static_cast<const float*>(A.gpu_data),
        static_cast<const float*>(B.gpu_data),
        static_cast<float*>(C.gpu_data),
        M, N, K,
        alpha, beta);

    cudaDeviceSynchronize();
}

// ============================================================================
// Host-side helper: bias + activation wrapper
// ============================================================================
static void convolution_cuda_bias_activation(
    CudaMat& output_blob,
    const CudaMat& bias_blob,
    int bias_term,
    int activation_type,
    const Mat& activation_params)
{
    int outch = output_blob.c;
    int outh = output_blob.h;
    int outw = output_blob.w;
    int out_cstep = (int)output_blob.cstep;
    int total = outch * outh * outw;

    float act_param0 = 0.0f;
    float act_param1 = 0.0f;
    if (activation_type == 2)
        act_param0 = activation_params[0];
    else if (activation_type == 3)
    {
        act_param0 = activation_params[0];
        act_param1 = activation_params[1];
    }
    else if (activation_type == 6)
    {
        act_param0 = activation_params[0];
        act_param1 = activation_params[1];
    }

    int threads_per_block = 256;
    int blocks_per_grid = (total + threads_per_block - 1) / threads_per_block;

    bias_activation_kernel<<<blocks_per_grid, threads_per_block>>>(
        static_cast<float*>(output_blob.gpu_data),
        static_cast<const float*>(bias_blob.gpu_data),
        outch, outh, outw, out_cstep,
        bias_term, activation_type,
        act_param0, act_param1);

    cudaDeviceSynchronize();
}

// ============================================================================
// Host-side helper: copy GEMM output to ncnn output layout
// GEMM output: [M, N] row-major continuous (M=outch, N=outh*outw)
// ncnn output: [outch, outh, outw] with cstep channel stride
// ============================================================================
static void convolution_cuda_copy_gemm_to_output(
    const float* gemm_out,
    float* output,
    int outch,
    int outh,
    int outw,
    int out_cstep)
{
    int N = outh * outw;
    for (int oc = 0; oc < outch; oc++)
    {
        cudaMemcpy2D(
            output + oc * out_cstep,
            outw * sizeof(float),
            gemm_out + oc * N,
            outw * sizeof(float),
            outw * sizeof(float),
            outh,
            cudaMemcpyDeviceToDevice);
    }
}

// ============================================================================
// Main convolution forward function
// Approach: im2col + GEMM + bias/activation
// ============================================================================
int Convolution_cuda::Convolution_cuda_forward(
    const CudaMat& input_blob,
    CudaMat& output_blob,
    const Option& opt) const
{
    // ---- Validate input ----
    if (input_blob.empty())
    {
        NCNN_LOGE("===Convolution_cuda_forward=== input_blob is empty");
        return -100;
    }

    // ---- Determine input dimensions ----
    int inch = 1;
    int inh = 1;
    int inw = input_blob.w;
    int batch = 1;

    if (input_blob.dims == 2)
    {
        inh = input_blob.h;
        inw = input_blob.w;
    }
    else if (input_blob.dims == 3)
    {
        inch = input_blob.c;
        inh = input_blob.h;
        inw = input_blob.w;
    }
    else if (input_blob.dims == 4)
    {
        batch = input_blob.d;
        inch = input_blob.c;
        inh = input_blob.h;
        inw = input_blob.w;
    }
    // dims == 1: inch=1, inh=1, inw=input_blob.w

    // ---- GEMM dimensions: compute K and verify inch ----
    int K = weight_data_size / num_output;                 // K = IC * KH * KW
    int M = num_output;                                     // M = OC

    // Override inch from weight dimensions for correctness (handles edge cases)
    int inch_from_weight = K / (kernel_h * kernel_w);
    if (inch_from_weight > 0 && inch_from_weight != inch)
    {
        NCNN_LOGE("  *  Convolution_cuda_forward: inch from input=%d, inch from weight=%d, using weight",
                  inch, inch_from_weight);
        inch = inch_from_weight;
    }

    // ---- Handle SAME padding (-233 = SAME_UPPER, -234 = SAME_LOWER) ----
    int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    int eff_pad_left = pad_left;
    int eff_pad_right = pad_right;
    int eff_pad_top = pad_top;
    int eff_pad_bottom = pad_bottom;
    int outw, outh;

    if (pad_left == -233 && pad_right == -233 && pad_top == -233 && pad_bottom == -233)
    {
        // SAME_UPPER: output size = ceil(input / stride)
        outw = (inw + stride_w - 1) / stride_w;
        outh = (inh + stride_h - 1) / stride_h;
        int wpad = (outw - 1) * stride_w + kernel_extent_w - inw;
        int hpad = (outh - 1) * stride_h + kernel_extent_h - inh;
        if (wpad < 0) wpad = 0;
        if (hpad < 0) hpad = 0;
        eff_pad_left = wpad / 2;
        eff_pad_right = wpad - wpad / 2;
        eff_pad_top = hpad / 2;
        eff_pad_bottom = hpad - hpad / 2;
        NCNN_LOGE("  *  SAME_UPPER: out=%dx%d wpad=%d hpad=%d pad=(%d,%d,%d,%d)",
                  outw, outh, wpad, hpad, eff_pad_left, eff_pad_right, eff_pad_top, eff_pad_bottom);
    }
    else if (pad_left == -234 && pad_right == -234 && pad_top == -234 && pad_bottom == -234)
    {
        // SAME_LOWER: output size = ceil(input / stride)
        outw = (inw + stride_w - 1) / stride_w;
        outh = (inh + stride_h - 1) / stride_h;
        int wpad = (outw - 1) * stride_w + kernel_extent_w - inw;
        int hpad = (outh - 1) * stride_h + kernel_extent_h - inh;
        if (wpad < 0) wpad = 0;
        if (hpad < 0) hpad = 0;
        eff_pad_left = wpad - wpad / 2;
        eff_pad_right = wpad / 2;
        eff_pad_top = hpad - hpad / 2;
        eff_pad_bottom = hpad / 2;
        NCNN_LOGE("  *  SAME_LOWER: out=%dx%d wpad=%d hpad=%d pad=(%d,%d,%d,%d)",
                  outw, outh, wpad, hpad, eff_pad_left, eff_pad_right, eff_pad_top, eff_pad_bottom);
    }
    else
    {
        // Explicit padding
        outw = (inw + pad_left + pad_right - kernel_extent_w) / stride_w + 1;
        outh = (inh + pad_top + pad_bottom - kernel_extent_h) / stride_h + 1;
    }

    if (outw <= 0 || outh <= 0)
    {
        NCNN_LOGE("===Convolution_cuda_forward=== invalid output size: outw=%d outh=%d", outw, outh);
        return -100;
    }

    NCNN_LOGE("  *  Convolution_cuda_forward: inch=%d outch=%d in=%dx%d out=%dx%d kernel=%dx%d K=%d",
              inch, num_output, inw, inh, outw, outh, kernel_w, kernel_h, K);

    int N_spatial = outh * outw;                            // N = OH * OW

    // ---- Allocate output tensor (always channeled, like ncnn CPU path) ----
    size_t elemsize = input_blob.elemsize;

    if (input_blob.dims == 4)
    {
        output_blob.create(outw, outh, batch, M, elemsize);
    }
    else
    {
        output_blob.create(outw, outh, M, elemsize);
    }

    if (output_blob.empty())
    {
        NCNN_LOGE("===Convolution_cuda_forward=== failed to allocate output_blob");
        return -100;
    }

    // ---- Raw pointers and strides ----
    const float* input_data = static_cast<const float*>(input_blob.gpu_data);
    float* output_data = static_cast<float*>(output_blob.gpu_data);
    int in_cstep = (int)input_blob.cstep;
    int out_cstep = (int)output_blob.cstep;

    // ---- Process each batch ----
    int total_batch = (input_blob.dims == 4) ? batch : 1;

    for (int b = 0; b < total_batch; b++)
    {
        const float* input_b = input_data;
        float* output_b = output_data;

        if (input_blob.dims == 4)
        {
            input_b += b * inch * in_cstep;
            output_b += b * M * out_cstep;
        }

        // ---- Step 1: im2col ----
        CudaMat col_blob;
        convolution_cuda_im2col_raw(
            input_b,
            col_blob,
            inch,
            inh, inw, in_cstep,
            kernel_w, kernel_h,
            stride_w, stride_h,
            dilation_w, dilation_h,
            eff_pad_left, eff_pad_top,
            outw, outh);

        if (col_blob.empty())
        {
            NCNN_LOGE("===Convolution_cuda_forward=== im2col failed for batch %d", b);
            return -100;
        }

        // ---- Step 2: GEMM  C[M][N] = A[M][K] * B[K][N] ----
        CudaMat gemm_out;
        gemm_out.create(N_spatial, M, elemsize);   // w=N, h=M → [M, N] row-major
        if (gemm_out.empty())
        {
            NCNN_LOGE("===Convolution_cuda_forward=== gemm_out allocation failed");
            col_blob.release();
            return -100;
        }

        convolution_cuda_gemm(
            weight_blob,    // A: M x K
            col_blob,       // B: K x N
            gemm_out,       // C: M x N
            M, N_spatial, K,
            0, 0);

        // ---- Step 3: Copy GEMM result to ncnn output layout ----
        convolution_cuda_copy_gemm_to_output(
            static_cast<const float*>(gemm_out.gpu_data),
            output_b,
            M, outh, outw, out_cstep);

        // Clean up
        col_blob.release();
        gemm_out.release();
    }

    // ---- Step 4: Bias + Activation (apply once on the full output) ----
    convolution_cuda_bias_activation(
        output_blob,
        bias_blob,
        bias_term,
        activation_type,
        activation_params);

    return 0;
}

} // namespace ncnn
