//
// Created by GIBEREZ on 2025/12/31.
//
// Pure-CUDA ArgMax / Top-K implementation.
// Strategy (mirrors the CPU version's partial_sort approach):
//   1. Each thread scans its chunk, keeping a local sorted top-k in registers.
//   2. Warp-level merge combines per-thread top-k via warp shuffles.
//   3. Block-level merge combines per-warp top-k via shared memory.
//   4. Each block writes its top-k to a global temp buffer.
//   5. If num_blocks == 1  →  write directly to output.
//      If num_blocks >  1  →  launch a single-block merge kernel.
//

#include "argmax_cuda.h"

#define ARGMAX_MAX_K 32

namespace ncnn {

// ---------------------------------------------------------------------------
// Device helper: insert (val, idx) into a descending-sorted top-k array.
// vals[0] is the largest; vals[k-1] is the smallest among the top-k.
// If val is not larger than vals[k-1], it is ignored.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void insert_topk(
    float val, int idx, float* vals, int* idxs, int k)
{
    // Find the first position where vals[pos] < val
    int pos = 0;
    while (pos < k && vals[pos] >= val) pos++;

    if (pos >= k) return;  // not in top-k

    // Shift smaller elements one position to the right
    for (int i = k - 1; i > pos; i--)
    {
        vals[i] = vals[i - 1];
        idxs[i] = idxs[i - 1];
    }
    vals[pos] = val;
    idxs[pos] = idx;
}

// ---------------------------------------------------------------------------
// Device helper: merge two descending-sorted top-k arrays (src → dst).
// ---------------------------------------------------------------------------
__device__ __forceinline__ void merge_topk(
    float* dst_vals, int* dst_idxs,
    const float* src_vals, const int* src_idxs, int k)
{
    float merged_vals[ARGMAX_MAX_K];
    int   merged_idxs[ARGMAX_MAX_K];

    int i = 0, j = 0, m = 0;
    while (m < k && (i < k || j < k))
    {
        float vi = (i < k) ? dst_vals[i] : -1e38f;
        float vj = (j < k) ? src_vals[j] : -1e38f;
        if (vi >= vj)
        {
            merged_vals[m] = dst_vals[i];
            merged_idxs[m] = dst_idxs[i];
            i++;
        }
        else
        {
            merged_vals[m] = src_vals[j];
            merged_idxs[m] = src_idxs[j];
            j++;
        }
        m++;
    }

    for (int x = 0; x < k; x++)
    {
        dst_vals[x] = merged_vals[x];
        dst_idxs[x] = merged_idxs[x];
    }
}

// ===================================================================
// Kernel A: per-block top-k discovery
// ===================================================================
__global__ void argmax_block_topk_kernel(
    const float* __restrict__ input,
    int                      total_size,
    float*                   block_vals,   // [num_blocks * k]
    float*                   block_idxs,   // [num_blocks * k]
    int                      k)
{
    // ---- per-thread local top-k (registers) ----
    float t_vals[ARGMAX_MAX_K];
    int   t_idxs[ARGMAX_MAX_K];
    for (int i = 0; i < k; i++)
    {
        t_vals[i] = -1e38f;
        t_idxs[i] = -1;
    }

    // ---- each thread scans its strided chunk ----
    for (int i = threadIdx.x + blockIdx.x * blockDim.x;
         i < total_size;
         i += blockDim.x * gridDim.x)
    {
        insert_topk(input[i], i, t_vals, t_idxs, k);
    }

    // ---- warp-level reduction (first lane collects from the warp) ----
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
    {
        // Exchange top-k via warp shuffles, one element at a time
        for (int e = 0; e < k; e++)
        {
            float peer_val = __shfl_down_sync(0xffffffff, t_vals[e], offset);
            int   peer_idx = __shfl_down_sync(0xffffffff, t_idxs[e], offset);
            insert_topk(peer_val, peer_idx, t_vals, t_idxs, k);
        }
    }

    // ---- block-level reduction via shared memory ----
    __shared__ float s_vals[ARGMAX_MAX_K * 32];  // up to 32 warps per block
    __shared__ int   s_idxs[ARGMAX_MAX_K * 32];

    int lane_id   = threadIdx.x % warpSize;
    int warp_id   = threadIdx.x / warpSize;
    int num_warps = blockDim.x / warpSize;

    if (lane_id == 0)
    {
        // First lane of each warp writes to shared memory
        float* wp_vals = s_vals + warp_id * k;
        int*   wp_idxs = s_idxs + warp_id * k;
        for (int e = 0; e < k; e++)
        {
            wp_vals[e] = t_vals[e];
            wp_idxs[e] = t_idxs[e];
        }
    }
    __syncthreads();

    // Warp 0, lane 0 merges all warp-level results into final block top-k
    if (warp_id == 0 && lane_id == 0)
    {
        float b_vals[ARGMAX_MAX_K];
        int   b_idxs[ARGMAX_MAX_K];
        for (int e = 0; e < k; e++)
        {
            b_vals[e] = -1e38f;
            b_idxs[e] = -1;
        }

        for (int w = 0; w < num_warps; w++)
        {
            float* wp_vals = s_vals + w * k;
            int*   wp_idxs = s_idxs + w * k;
            for (int e = 0; e < k; e++)
            {
                insert_topk(wp_vals[e], wp_idxs[e], b_vals, b_idxs, k);
            }
        }

        // Write block result to global memory
        float* out_vals = block_vals + blockIdx.x * k;
        float* out_idxs = block_idxs + blockIdx.x * k;
        for (int e = 0; e < k; e++)
        {
            out_vals[e] = b_vals[e];
            out_idxs[e] = static_cast<float>(b_idxs[e]);
        }
    }
}

// ===================================================================
// Kernel B: merge block-level results into final output
// ===================================================================
__global__ void argmax_merge_blocks_kernel(
    const float* block_vals,    // [num_blocks * k]
    const float* block_idxs,    // [num_blocks * k]
    int          num_blocks,
    float*       out_vals,      // [k]
    float*       out_idxs,      // [k]
    int          k)
{
    // ---- per-thread local top-k (registers) ----
    float t_vals[ARGMAX_MAX_K];
    int   t_idxs[ARGMAX_MAX_K];
    for (int e = 0; e < k; e++)
    {
        t_vals[e] = -1e38f;
        t_idxs[e] = -1;
    }

    // Each thread merges a strided subset of the block-level results
    int total = num_blocks * k;
    for (int i = threadIdx.x + blockIdx.x * blockDim.x;
         i < total;
         i += blockDim.x * gridDim.x)
    {
        float val = block_vals[i];
        int   idx = static_cast<int>(block_idxs[i]);
        insert_topk(val, idx, t_vals, t_idxs, k);
    }

    // ---- warp-level reduction ----
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
    {
        for (int e = 0; e < k; e++)
        {
            float peer_val = __shfl_down_sync(0xffffffff, t_vals[e], offset);
            int   peer_idx = __shfl_down_sync(0xffffffff, t_idxs[e], offset);
            insert_topk(peer_val, peer_idx, t_vals, t_idxs, k);
        }
    }

    // ---- block-level reduction via shared memory ----
    __shared__ float s_vals[ARGMAX_MAX_K * 32];
    __shared__ int   s_idxs[ARGMAX_MAX_K * 32];

    int lane_id   = threadIdx.x % warpSize;
    int warp_id   = threadIdx.x / warpSize;
    int num_warps = blockDim.x / warpSize;

    if (lane_id == 0)
    {
        float* wp_vals = s_vals + warp_id * k;
        int*   wp_idxs = s_idxs + warp_id * k;
        for (int e = 0; e < k; e++)
        {
            wp_vals[e] = t_vals[e];
            wp_idxs[e] = t_idxs[e];
        }
    }
    __syncthreads();

    // Warp 0, lane 0 merges all warp results
    if (warp_id == 0 && lane_id == 0)
    {
        float m_vals[ARGMAX_MAX_K];
        int   m_idxs[ARGMAX_MAX_K];
        for (int e = 0; e < k; e++)
        {
            m_vals[e] = -1e38f;
            m_idxs[e] = -1;
        }

        for (int w = 0; w < num_warps; w++)
        {
            float* wp_vals = s_vals + w * k;
            int*   wp_idxs = s_idxs + w * k;
            for (int e = 0; e < k; e++)
            {
                insert_topk(wp_vals[e], wp_idxs[e], m_vals, m_idxs, k);
            }
        }

        for (int e = 0; e < k; e++)
        {
            out_vals[e] = m_vals[e];
            out_idxs[e] = static_cast<float>(m_idxs[e]);
        }
    }
}

// ===================================================================
// Host-side entry point
// ===================================================================
int ArgMax_cuda::argmax_cuda(const CudaMat& input_blob, CudaMat& output_blob) const
{
    int size = input_blob.total();

    // Clamp topk to actual data size
    int k = topk;
    if (k > size)  k = size;
    if (k <= 0)    return -100;
    if (k > ARGMAX_MAX_K)  k = ARGMAX_MAX_K;   // hard limit

    // Allocate output blob
    // Layout: w = k, h = 1 (indices only) or h = 2 (values + indices)
    if (out_max_val)
        output_blob.create(k, 2, 4);
    else
        output_blob.create(k, 1, 4);
    if (output_blob.empty())
        return -100;

    const float* d_input  = static_cast<const float*>(input_blob.gpu_data);
    float*       d_output = static_cast<float*>(output_blob.gpu_data);

    // ---- Determine grid size ----
    int threadsPerBlock = 256;
    int num_blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    if (num_blocks > 1024)  num_blocks = 1024;   // reasonable cap

    if (num_blocks == 0)
        return -100;

    // ---- Temp storage for block-level results ----
    float* d_block_vals = nullptr;
    float* d_block_idxs = nullptr;
    if (num_blocks > 1)
    {
        cudaMalloc(&d_block_vals, num_blocks * k * sizeof(float));
        cudaMalloc(&d_block_idxs, num_blocks * k * sizeof(float));
    }

    // ---- Launch per-block top-k kernel ----
    argmax_block_topk_kernel<<<num_blocks, threadsPerBlock>>>(
        d_input, size,
        (num_blocks > 1) ? d_block_vals : d_output,
        (num_blocks > 1) ? d_block_idxs : (out_max_val ? (d_output + k) : d_output),
        k);
    cudaDeviceSynchronize();

    // ---- Merge phase (only if multiple blocks) ----
    if (num_blocks > 1)
    {
        // Merge: output the final top-k values and indices into temp
        // (the merge kernel writes values to d_block_vals[0..k-1] and indices to
        //  d_block_idxs[0..k-1] as scratch space)
        argmax_merge_blocks_kernel<<<1, 256>>>(
            d_block_vals, d_block_idxs, num_blocks,
            d_block_vals,  // reuse as output scratch: values
            d_block_idxs,  // reuse as output scratch: indices
            k);
        cudaDeviceSynchronize();

        // Copy merged results to actual output
        if (out_max_val)
        {
            cudaMemcpy(d_output,       d_block_vals, k * sizeof(float), cudaMemcpyDeviceToDevice);
            cudaMemcpy(d_output + k,   d_block_idxs, k * sizeof(float), cudaMemcpyDeviceToDevice);
        }
        else
        {
            cudaMemcpy(d_output,       d_block_idxs, k * sizeof(float), cudaMemcpyDeviceToDevice);
        }

        cudaFree(d_block_vals);
        cudaFree(d_block_idxs);
    }
    else
    {
        // Single block: result is already in d_output.
        // If out_max_val, the kernel wrote indices to d_output + k,
        // but we still need the values at d_output[0..k-1].
        // The kernel wrote both values and indices in the correct layout:
        //   block_vals → d_output (values)
        //   block_idxs → d_output + k (indices, if out_max_val)
        //   or d_output (indices only, if !out_max_val)
        // So for out_max_val, both values and indices are correctly placed.
        // For !out_max_val, indices are in d_output.  No extra copy needed.
    }

    cudaDeviceSynchronize();
    return 0;
}

} // namespace ncnn
