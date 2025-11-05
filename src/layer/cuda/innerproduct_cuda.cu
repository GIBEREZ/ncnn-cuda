#include <cuda_runtime.h>
#include "innerproduct_cuda.h"

namespace ncnn {
    __global__ void GEMM(const void* input_blob, void* output_blob)
    {
        // 计算全局线程索引（global thread index）
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x * 4;
        unsigned int idxElement = idx * 4;
        if (idxElement + 3 < number) {
            float4 vec_in = *(float4*)&input_blob[idxElement];
            float4 vec_out;
    }

    int InnerProduct_cuda::InnerProduct_cuda_forward(const CudaMat& input_blob, CudaMat& output_blob, const Option& opt) const
    {
        // 1.输入形状解析
        const int num_input = weight_data_size / num_output;

        int w = input_blob.w;
        int h = input_blob.h;
        int channels = input_blob.c;
        size_t elemsize = input_blob.elemsize;
        int size = w * h;

        if (input_blob.dims == 2 && w == num_input)
        {
            // gemm
            output_blob.create(num_output, h, elemsize);
        }
        return 0;
    }
} // namespace ncnn