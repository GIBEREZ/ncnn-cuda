//
// Created by GIBEREZ on 2026/1/1.
//
#include "cumulativesum_cuda.h"
#include <cub/cub.cuh>

namespace ncnn {
    __global__ void ReducePartSum_kernel(const float* input, float* part_output, size_t Number)
    {
        extern __shared__ float sum[];
        int tid = threadIdx.x;
        int lane = tid % warpSize;
        int warpId = tid / warpSize;

        unsigned int idx = blockIdx.x * blockDim.x + tid;
        unsigned int idxElement = idx * 4;

        float threadSum = 0.0f;
        if (idxElement < Number) {
            size_t remain = Number - idxElement;
            float4 vec_in = make_float4(0, 0, 0, 0);

            if (remain >= 4) {
#if __CUDA_ARCH__ >= 350
                vec_in = __ldg(reinterpret_cast<const float4*>(input + idxElement));
#else
                vec_in = *reinterpret_cast<const float4*>(input + idxElement);
#endif
            } else {
                if (remain > 0) vec_in.x = input[idxElement];
                if (remain > 1) vec_in.y = input[idxElement + 1];
                if (remain > 2) vec_in.z = input[idxElement + 2];
            }
            threadSum = vec_in.x + vec_in.y + vec_in.z + vec_in.w;
        }

        #pragma unroll
        for (int offset = warpSize/2; offset > 0; offset /= 2) {
            threadSum += __shfl_down_sync(0xFFFFFFFF, threadSum, offset);
        }

        if (lane == 0) {
            sum[warpId] = threadSum;
        }
        __syncthreads();

        if (warpId == 0) {
            float blockSum = (lane < (blockDim.x + warpSize - 1) / warpSize) ? sum[lane] : 0.0f;

            #pragma unroll
            for (int offset = warpSize/2; offset > 0; offset /= 2) {
                blockSum += __shfl_down_sync(0xFFFFFFFF, blockSum, offset);
            }

            if (tid == 0) {
                part_output[blockIdx.x] = blockSum;
            }
        }
    }

    __global__ void AddBlockPrefix_inplace_kernel(float* input, const float* part_output, size_t Number)
    {

        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int idxElement = idx * 4;

        if (idxElement >= Number)
            return;

        float base = (blockIdx.x == 0) ? 0.0f : part_output[blockIdx.x - 1];
        size_t remain = Number - idxElement;
        if (remain >= 4) {
            float4 vec_in;

            if ((reinterpret_cast<size_t>(input + idxElement) & 0xF) == 0) {
                vec_in = *reinterpret_cast<const float4*>(input + idxElement);
            } else {
                vec_in.x = input[idxElement];
                vec_in.y = input[idxElement + 1];
                vec_in.z = input[idxElement + 2];
                vec_in.w = input[idxElement + 3];
            }

            float4 vec_out;
            vec_out.x = vec_in.x;
            vec_out.y = vec_in.y + vec_out.x;
            vec_out.z = vec_in.z + vec_out.y;
            vec_out.w = vec_in.w + vec_out.z;

            vec_out.x += base;
            vec_out.y += base;
            vec_out.z += base;
            vec_out.w += base;

            if ((reinterpret_cast<size_t>(input + idxElement) & 0xF) == 0) {
                *reinterpret_cast<float4*>(input + idxElement) = vec_out;
            } else {
                input[idxElement] = vec_out.x;
                input[idxElement + 1] = vec_out.y;
                input[idxElement + 2] = vec_out.z;
                input[idxElement + 3] = vec_out.w;
            }
        } else {
            float partial_sum = 0.0f;
            for (int i = 0; i < remain; i++) {
                partial_sum += input[idxElement + i];
                input[idxElement + i] = partial_sum + base;
            }
        }
    }

    int cumulativesum_cuda_inplace(CudaMat& input_blob, int axis)
    {
        int dims = input_blob.dims;
        int positive_axis = axis < 0 ? dims + axis : axis;

        if (dims == 1)
        {
            void* d_temp_storage = nullptr;
            size_t temp_storage_bytes = 0;

            cub::DeviceScan::InclusiveSum(
                d_temp_storage,
                temp_storage_bytes,
                static_cast<float*>(input_blob.gpu_data),
                input_blob.total()
                );

            cudaMalloc(&d_temp_storage, temp_storage_bytes);

            cub::DeviceScan::InclusiveSum(
                d_temp_storage,
                temp_storage_bytes,
                static_cast<float*>(input_blob.gpu_data),
                input_blob.total()
                );

            return 0;
        }
        if (dims == 2 && positive_axis == 0)
        {
            int size = input_blob.w * input_blob.h;

            int threadsPerBlock = 256;
            int vec_size = 16 / input_blob.elemsize;
            int totalThreadsNeeded = (size + vec_size - 1) / vec_size;
            int blocksPerGrid = (totalThreadsNeeded + threadsPerBlock - 1) / threadsPerBlock;
            float* part_output = nullptr;
            cudaMalloc(&part_output, blocksPerGrid * sizeof(float));
            int warpSize;
            cudaDeviceGetAttribute(&warpSize, cudaDevAttrWarpSize, 0);
            int warps_per_block = (threadsPerBlock + warpSize - 1) / warpSize;
            size_t sharedBytes = warps_per_block * sizeof(float);
            ReducePartSum_kernel<<<blocksPerGrid, threadsPerBlock, sharedBytes>>>(static_cast<float*>(input_blob.gpu_data), part_output, size);
            cudaDeviceSynchronize();

            void* d_temp_storage = nullptr;
            size_t temp_storage_bytes = 0;
            cub::DeviceScan::InclusiveSum(
                d_temp_storage,
                temp_storage_bytes,
                part_output,
                part_output,
                blocksPerGrid
            );
            cudaMalloc(&d_temp_storage, temp_storage_bytes);
            cub::DeviceScan::InclusiveSum(
                d_temp_storage,
                temp_storage_bytes,
                part_output,
                part_output,
                blocksPerGrid
            );
            cudaFree(d_temp_storage);
            AddBlockPrefix_inplace_kernel<<<blocksPerGrid, threadsPerBlock>>>(static_cast<float*>(input_blob.gpu_data), part_output, size);
            cudaDeviceSynchronize();
        }
        if (dims == 2 && positive_axis == 1)
        {
            int w = input_blob.w;
            int h = input_blob.h;

            cudaStream_t* streams = new cudaStream_t[h];
            void** temp_storages = new void*[h];
            size_t temp_storage_bytes = 0;

            cub::DeviceScan::InclusiveSum(
                nullptr,
                temp_storage_bytes,
                static_cast<float*>(nullptr),
                static_cast<float*>(nullptr),
                w
            );

            for (int i = 0; i < h; ++i) {
                cudaStreamCreate(&streams[i]);
                cudaMalloc(&temp_storages[i], temp_storage_bytes);
            }

            for (int i = 0; i < h; ++i) {
                float* row_ptr = static_cast<float*>(input_blob.data) + i * w;

                cub::DeviceScan::InclusiveSum(
                    temp_storages[i],
                    temp_storage_bytes,
                    row_ptr,
                    row_ptr,
                    w,
                    streams[i]
                );
            }

            for (int i = 0; i < h; ++i) {
                cudaStreamSynchronize(streams[i]);
                cudaStreamDestroy(streams[i]);
                cudaFree(temp_storages[i]);
            }

            delete[] streams;
            delete[] temp_storages;
            return 0;
        }
        if (dims == 3 && positive_axis == 0)
        {
            int w = input_blob.w;
            int h = input_blob.h;
            int c = input_blob.c;
            int size = w * h;


        }

        return 0;
    }
}