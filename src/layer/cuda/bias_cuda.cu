#include "bias_cuda.h"
#include <cuda_runtime.h>
#include <cudnn.h>

namespace ncnn {
    int Bias_cuda::Bias_cuda_forward_inplace(CudaMat& input_blob) const
    {
        // 1.Create cuDNN handle
        cudnnHandle_t handle;
        if (cudnnCreate(&handle) != CUDNN_STATUS_SUCCESS) return -1;

        // 2. Create tensor descriptors (A, bias)
        cudnnTensorDescriptor_t A_desc,  bias_desc;

        // 3. Initialize these descriptors
        cudnnCreateTensorDescriptor(&A_desc);                       // Create input tensor descriptor
        cudnnCreateTensorDescriptor(&bias_desc);                    // Create bias tensor descriptor

        // 4. Set input/output shape descriptors
        cudnnDataType_t cudnn_dtype;
        if (input_blob.elemsize == 4) cudnn_dtype = CUDNN_DATA_FLOAT;
        else if (input_blob.elemsize == 2) cudnn_dtype = CUDNN_DATA_HALF;
        else if (input_blob.elemsize == 1)
        {
            NCNN_LOGE("===Bias_cuda::Bias_cuda_forward_inplace(CudaMat& input_blob)=== Int8 addition is not supported;");
            return -1;
        }
        else return -1;

        // 5. Set input output bias Tensor4d Descriptor
        int N=1, C=1, H=1, W=1;
        if (input_blob.dims == 1)
        {
            C = input_blob.w;
        }
        else if (input_blob.dims == 2)
        {
            C = input_blob.h;
            W = input_blob.w;
        }
        else if (input_blob.dims == 3)
        {
            C = input_blob.c;
            H = input_blob.h;
            W = input_blob.w;
        }
        else if (input_blob.dims == 4)
        {
            N = input_blob.d;
            C = input_blob.c;
            H = input_blob.h;
            W = input_blob.w;
        }
        cudnnSetTensor4dDescriptor(A_desc,  CUDNN_TENSOR_NCHW, cudnn_dtype, N, C, H, W);
        cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW, cudnn_dtype, 1, C, 1, 1);

        // 6.add Tensor
        float alpha = 1.0f;
        float beta = 1.0f;
        cudnnAddTensor(
            handle,
            &alpha,
            bias_desc,
            bias_blob.gpu_data,
            &beta,
            A_desc,
            input_blob.gpu_data
        );

        // 7.release
        cudnnDestroyTensorDescriptor(A_desc);
        cudnnDestroyTensorDescriptor(bias_desc);
        cudnnDestroy(handle);

        return 0;
    }

}