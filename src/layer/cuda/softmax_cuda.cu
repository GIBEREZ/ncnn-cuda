#include "softmax_cuda.h"
#include "system.cu"
#include "system.h"

#include <cuda_runtime.h>

namespace ncnn {
    template<typename LOAD, typename STORE, int PACK_SIZE, int BLOCK_SIZE>
    __global__ void softmax_kernel(LOAD load, STORE store, int width)
    {
        extern __shared__ float block_warp_max[];
        int row = blockIdx.x;                        // BlockIdx.x corresponds to batch dimension N
        int tid = threadIdx.x;                       // The ID of the thread within the program block
        int lane = tid & 31;                         // warp lane id
        int warp_id = tid >> 5;                      // The warp ID (within the block) to which this thread belongs
        int num_warps = (BLOCK_SIZE + 31) / 32;

        float vec[PACK_SIZE];
        float vm = -FLT_MAX;
        for (int idx = tid * PACK_SIZE; idx < width; idx += BLOCK_SIZE * PACK_SIZE)
        {
            if (idx + PACK_SIZE <= width)
                load.template load<PACK_SIZE>(vec, row, idx);
            else
            {
                for (int i = 0; i < PACK_SIZE; ++i)
                    vec[i] = (idx + i < width) ? load.template load<1>(&vec[i], row, idx + i), vec[i] : -FLT_MAX;
            }
            vm = fmaxf(fmaxf(vec[0], vec[1]), fmaxf(vec[2], vec[3]));
        }

        for (int offset = 16; offset > 0; offset >>= 1)
        {
            float other = __shfl_down_sync(0xffffffff, vm, offset);
            vm = fmaxf(vm, other);
        }

        if (lane == 0) block_warp_max[warp_id] = vm;
        __syncthreads();

        float thread_max = -FLT_MAX;
        if (tid < num_warps) {
            thread_max = block_warp_max[tid];
            for (int offset = 16; offset > 0; offset >>= 1) {
                float other = __shfl_down_sync(0xffffffff, thread_max, offset);
                thread_max = fmaxf(thread_max, other);
            }
        }

        float block_max = block_warp_max[0];
        if (tid == 0) block_warp_max[0] = thread_max;
        __syncthreads();
        block_max = block_warp_max[0];

        int idx = tid * PACK_SIZE;
        #pragma unroll
        for (int i = 0; i < PACK_SIZE; ++i) {
            if (idx + i < width) vec[i] = expf(vec[i] - block_max);
        }

        float thread_sum = 0;
        #pragma unroll
        for (int i = 0; i < PACK_SIZE; ++i) {
            if (idx + i < width) thread_sum += vec[i];
        }

        for (int offset = 16; offset > 0; offset >>= 1) {
            float other = __shfl_down_sync(0xffffffff, thread_sum, offset);
            thread_sum += other;
        }

        if (lane == 0) block_warp_max[warp_id] = thread_sum;
        __syncthreads();

        if (tid < num_warps) {
            thread_sum = block_warp_max[tid];
            for (int offset = 16; offset > 0; offset >>= 1) {
                float other = __shfl_down_sync(0xffffffff, thread_sum, offset);
                thread_sum += other;
            }
        }

        float block_sum = block_warp_max[0];
        if (tid == 0) block_warp_max[0] = thread_sum;
        __syncthreads();
        block_sum = block_warp_max[0];

        store.template store<PACK_SIZE>(vec, row, idx);
    }

    void softmax_cuda(const CudaMat& input_blob, CudaMat& output_blob)
    {
        int width = input_blob.w * input_blob.h * input_blob.c;
        const int BLOCK_SIZE = 1024;
        dim3 grid(input_blob.h * input_blob.c);
        if (input_blob.elemsize == 4)
        {
            LOAD load(input_blob.gpu_data, input_blob.w);
            STORE store(output_blob.gpu_data, output_blob.w);
            const int PACK_SIZE = 4;

            softmax_kernel<LOAD, STORE, PACK_SIZE, BLOCK_SIZE>
                <<<grid, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(load, store, width);
        }
        else if (input_blob.elemsize == 2)
        {

        }
        else if (input_blob.elemsize == 1)
        {

        }
    }

    void softmax_cuda_inplace(CudaMat& input_blob)
    {
        if (input_blob.elemsize == 4)
        {

        }
        else if (input_blob.elemsize == 2)
        {

        }
        else if (input_blob.elemsize == 1)
        {

        }
    }
}
