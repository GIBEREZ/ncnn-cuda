#include "cast_cuda.h"
#include "system.h"

namespace ncnn {
    int Cast_cuda::Cast_cuda_forward(const CudaMat& input_blob, CudaMat& output_blob) const
    {
        int w = input_blob.w;
        int h = input_blob.h;
        int d = input_blob.d;
        int channels = input_blob.c;
        int dims = input_blob.dims;
        size_t elemsize = input_blob.elemsize;
        size_t out_elemsize = elemsize;

        if (type_to == 1) out_elemsize = 4;
        else if (type_to == 2) out_elemsize = 2;
        else if (type_to == 3) out_elemsize = 1;

        if (dims == 1) output_blob.create(w, out_elemsize);
        else if (dims == 2) output_blob.create(w, h, out_elemsize);
        else if (dims == 3) output_blob.create(w, h, channels, out_elemsize);
        else if (dims == 4) output_blob.create(w, h, d, channels, out_elemsize);
        if (output_blob.empty()) return -100;

        if (type_from == 1 && type_to == 2)
        {
            int number = input_blob.total();
            int threadsPerBlock = 256;
            int blocksPerGrid = (number + threadsPerBlock - 1) / threadsPerBlock;

            Cast_kernel_FP32_to_FP16<<<blocksPerGrid, threadsPerBlock>>>(
                static_cast<float*>(input_blob.gpu_data),
                static_cast<half*>(output_blob.gpu_data),
                number
            );
            cudaDeviceSynchronize();
        }
        else if (type_from == 1 && type_to == 3)
        {
            int number = input_blob.total();
            int threadsPerBlock = 256;
            int blocksPerGrid = (number + threadsPerBlock - 1) / threadsPerBlock;

            Cast_kernel_FP32_to_INT8<<<blocksPerGrid, threadsPerBlock>>>(
                static_cast<float*>(input_blob.gpu_data),
                static_cast<int8_t*>(output_blob.gpu_data),
                number
            );
            cudaDeviceSynchronize();
        }
        else if (type_from == 2 && type_to == 1)
        {
            int number = input_blob.total();
            int threadsPerBlock = 256;
            int blocksPerGrid = (number + threadsPerBlock - 1) / threadsPerBlock;

            Cast_kernel_FP16_to_FP32<<<blocksPerGrid, threadsPerBlock>>>(
                static_cast<half*>(input_blob.gpu_data),
                static_cast<float*>(output_blob.gpu_data),
                number
            );
            cudaDeviceSynchronize();
        }
        else if (type_from == 2 && type_to == 3)
        {
            int number = input_blob.total();
            int threadsPerBlock = 256;
            int blocksPerGrid = (number + threadsPerBlock - 1) / threadsPerBlock;

            Cast_kernel_FP16_to_INT8<<<blocksPerGrid, threadsPerBlock>>>(
                static_cast<half*>(input_blob.gpu_data),
                static_cast<int8_t*>(output_blob.gpu_data),
                number
            );
            cudaDeviceSynchronize();
        }
        else if (type_from == 3 && type_to == 1)
        {
            int number = input_blob.total();
            int threadsPerBlock = 256;
            int blocksPerGrid = (number + threadsPerBlock - 1) / threadsPerBlock;

            Cast_kernel_INT8_to_FP32<<<blocksPerGrid, threadsPerBlock>>>(
                static_cast<int8_t*>(input_blob.gpu_data),
                static_cast<float*>(output_blob.gpu_data),
                number
            );
            cudaDeviceSynchronize();
        }
        else if (type_from == 3 && type_to == 2)
        {
            int number = input_blob.total();
            int threadsPerBlock = 256;
            int blocksPerGrid = (number + threadsPerBlock - 1) / threadsPerBlock;

            Cast_kernel_INT8_to_FP16<<<blocksPerGrid, threadsPerBlock>>>(
                static_cast<int8_t*>(input_blob.gpu_data),
                static_cast<half*>(output_blob.gpu_data),
                number
            );
            cudaDeviceSynchronize();
        }

        return 0;
    }
} // namespace ncnn