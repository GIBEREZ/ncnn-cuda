//
// Created by GIBEREZ on 2025/11/5.
// CUDA custom InnerProduct (Fully-Connected) implementation
//
#include "innerproduct_cuda.h"
#include "system.h"
#include <cuda_runtime.h>
#include <cfloat>

namespace ncnn {

// ============================================================================
// Kernel 1: GEMM for InnerProduct
// C[M][N] = A[M][K] * W[N][K]^T
// A = input   [M][K] row-major
// W = weight  [N][K] row-major (stored as [num_output][num_input])
// C = output  [M][N] row-major
// Computation: C[m][n] = sum_k A[m][k] * W[n][k]
// ============================================================================
#define TILE_M 16
#define TILE_N 16
#define TILE_K 16

__global__ void innerproduct_gemm_kernel(
    const float* A,
    const float* W,
    float* C,
    int M,
    int N,
    int K)
{
    int row = blockIdx.y * TILE_M + threadIdx.y;
    int col = blockIdx.x * TILE_N + threadIdx.x;

    __shared__ float As[TILE_M][TILE_K];
    __shared__ float Ws[TILE_N][TILE_K];

    float sum = 0.0f;

    for (int k_block = 0; k_block < (K + TILE_K - 1) / TILE_K; k_block++)
    {
        // Load tile of A: As[ty][tx] = A[row][k_block*TILE_K + tx]
        int a_k = k_block * TILE_K + threadIdx.x;
        if (row < M && a_k < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + a_k];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        // Load tile of W: Ws[tx][ty] = W[col][k_block*TILE_K + ty]
        int w_k = k_block * TILE_K + threadIdx.y;
        if (col < N && w_k < K)
            Ws[threadIdx.x][threadIdx.y] = W[col * K + w_k];
        else
            Ws[threadIdx.x][threadIdx.y] = 0.0f;

        __syncthreads();

        // Dot product: sum += As[ty][k] * Ws[tx][k]
        #pragma unroll
        for (int k = 0; k < TILE_K; k++)
        {
            sum += As[threadIdx.y][k] * Ws[threadIdx.x][k];
        }

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}

// ============================================================================
// Kernel 2: Bias + Activation (fused)
// ============================================================================
__global__ void innerproduct_bias_activation_kernel(
    float* output,
    const float* bias,
    int M,
    int N,
    int bias_term,
    int activation_type,
    float act_param0,
    float act_param1)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N;
    if (idx >= total)
        return;

    float val = output[idx];

    if (bias_term)
        val += bias[idx % N];

    switch (activation_type)
    {
    case 0: break;
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
            if (val < lower)      val = 0.0f;
            else if (val > upper) ;
            else                  val = val * (val * alpha + beta);
        }
        break;
    default:
        break;
    }

    output[idx] = val;
}

// ============================================================================
// Main InnerProduct forward
// ============================================================================
int InnerProduct_cuda::InnerProduct_cuda_forward(
    const CudaMat& input_blob,
    CudaMat& output_blob,
    const Option& opt) const
{
    if (input_blob.empty())
    {
        NCNN_LOGE("===InnerProduct_cuda_forward=== input_blob is empty");
        return -100;
    }

    // ---- Compute dimensions ----
    int K = weight_data_size / num_output;   // num_input
    int N = num_output;                       // num_output

    int w = input_blob.w;
    int h = input_blob.h;
    int channels = input_blob.c;
    int size = w * h;

    int M;  // batch size

    // Match ncnn CPU InnerProduct logic:
    //   dims=2 && w == K  →  [K, M] batched input, M = h
    //   otherwise          →  single flat vector, M = 1, K = total elements
    if (input_blob.dims == 2 && w == K)
    {
        M = h;
    }
    else
    {
        M = 1;
        if (channels > 0)
            K = size * channels;
        else
            K = w;
    }

    NCNN_LOGE("  *  InnerProduct_cuda_forward: M=%d K=%d N=%d  in_dims=%d w=%d h=%d c=%d",
              M, K, N, input_blob.dims, w, h, channels);

    // ---- Allocate output (match ncnn CPU convention) ----
    if (input_blob.dims == 2 && w == K)
    {
        // Batched: output [N, M]
        output_blob.create(N, M, input_blob.elemsize);
    }
    else
    {
        // Single vector: output [N]
        output_blob.create(N, input_blob.elemsize);
    }

    if (output_blob.empty())
    {
        NCNN_LOGE("===InnerProduct_cuda_forward=== output_blob allocation failed");
        return -100;
    }

    // ---- Run GEMM (FP32 only) ----
    size_t elemsize = input_blob.elemsize;

    if (elemsize == 4)
    {
        dim3 block_dim(TILE_N, TILE_M);
        dim3 grid_dim(
            (N + TILE_N - 1) / TILE_N,
            (M + TILE_M - 1) / TILE_M);

        innerproduct_gemm_kernel<<<grid_dim, block_dim>>>(
            static_cast<const float*>(input_blob.gpu_data),
            static_cast<const float*>(weight_blob.gpu_data),
            static_cast<float*>(output_blob.gpu_data),
            M, N, K);

        cudaDeviceSynchronize();
    }
    else
    {
        NCNN_LOGE("===InnerProduct_cuda_forward=== unsupported elemsize=%zu", elemsize);
        return -100;
    }

    // ---- Bias + Activation ----
    if (bias_term || activation_type != 0)
    {
        float act_param0 = 0.0f;
        float act_param1 = 0.0f;
        if (activation_type == 2)
            act_param0 = activation_params[0];
        else if (activation_type == 3 || activation_type == 6)
        {
            act_param0 = activation_params[0];
            act_param1 = activation_params[1];
        }

        int total = M * N;
        int threads_per_block = 256;
        int blocks_per_grid = (total + threads_per_block - 1) / threads_per_block;

        innerproduct_bias_activation_kernel<<<blocks_per_grid, threads_per_block>>>(
            static_cast<float*>(output_blob.gpu_data),
            static_cast<const float*>(bias_blob.gpu_data),
            M, N,
            bias_term, activation_type,
            act_param0, act_param1);

        cudaDeviceSynchronize();
    }

    return 0;
}

} // namespace ncnn