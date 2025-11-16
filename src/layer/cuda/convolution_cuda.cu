//
// Created by GIBEREZ on 2025/11/4.
//
#include "convolution_cuda.h"
#include <cuda_runtime.h>
#include <cudnn.h>

namespace ncnn {
    int Convolution_cuda::Convolution_cuda_forward(const CudaMat& input_blob, CudaMat& output_blob, const Option& opt) const
    {
        // 1. Create cuDNN handle
        cudnnHandle_t handle;
        if (cudnnCreate(&handle) != CUDNN_STATUS_SUCCESS) return -1;

        // 2. Create tensor descriptors (input, weight, output)
        cudnnTensorDescriptor_t input_desc, output_desc, bias_desc;
        cudnnFilterDescriptor_t filter_desc;
        cudnnConvolutionDescriptor_t conv_desc;
        cudnnActivationDescriptor_t activatio_desc;

        // 3. Initialize these descriptors
        cudnnCreateTensorDescriptor(&input_desc);                   // Create input tensor descriptor
        cudnnCreateTensorDescriptor(&output_desc);                  // Create output tensor descriptor
        cudnnCreateFilterDescriptor(&filter_desc);                  // Create filter descriptor
        cudnnCreateConvolutionDescriptor(&conv_desc);               // Create convolution descriptor
        cudnnCreateTensorDescriptor(&bias_desc);                    // Create bias tensor descriptor
        cudnnCreateActivationDescriptor(&activatio_desc);           // Create activation function descriptor

        // 4. Set input/output shape descriptors
        cudnnDataType_t cudnn_dtype;
        if (input_blob.elemsize == 4)
        {
            cudnn_dtype = CUDNN_DATA_FLOAT;
        }
        else if (input_blob.elemsize == 2)
        {
            cudnn_dtype = CUDNN_DATA_HALF;
        }
        else if (input_blob.elemsize == 1)
        {
            cudnn_dtype = CUDNN_DATA_INT8;
        }
        else
        {
            return -1;
        }

        int out_h = (input_blob.h + pad_top + pad_bottom - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
        int out_w = (input_blob.w + pad_left + pad_right - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

        if (input_blob.dims == 1)
        {
            output_blob.create(out_w, input_blob.elemsize);
            cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW,
                                       cudnn_dtype, 1, 1, 1, input_blob.w);
        }
        else if (input_blob.dims == 2)
        {
            output_blob.create(out_w, out_h, input_blob.elemsize);
            cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW,
                                       cudnn_dtype, 1, 1, input_blob.h, input_blob.w);
        }
        else if (input_blob.dims == 3)
        {
            output_blob.create(out_w, out_h, num_output, input_blob.elemsize);
            cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW,
                                       cudnn_dtype, 1, input_blob.c, input_blob.h, input_blob.w);
        }
        else if (input_blob.dims == 4)
        {
            output_blob.create(out_w, out_h, input_blob.d, num_output, input_blob.elemsize);
            cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW,
                                       cudnn_dtype, input_blob.d, input_blob.c, input_blob.h, input_blob.w);
        }
        else
        {
            return -1;
        }

        cudnnSetTensor4dDescriptor(output_desc,
            CUDNN_TENSOR_NCHW, cudnn_dtype,
            input_blob.d,
            num_output,
            out_h, out_w
        );

        // 5. Set filter descriptor
        cudnnSetFilter4dDescriptor(filter_desc, cudnn_dtype,
                                   CUDNN_TENSOR_NCHW,
                                   num_output,
                                   weight_data_size / (num_output * kernel_h * kernel_w),
                                   kernel_h,
                                   kernel_w
        );

        // 6. Set convolution parameters (padding, stride, dilation); cuDNN does not support asymmetric padding
        cudnnSetConvolution2dDescriptor(conv_desc,
                                        std::max(pad_top, pad_bottom), std::max(pad_left, pad_right),
                                        stride_h, stride_w,
                                        dilation_h, dilation_w,
                                        CUDNN_CROSS_CORRELATION,
                                        cudnn_dtype
        );

        // 7. Compute workspace size
        size_t workspace_bytes = 0;
        cudnnGetConvolutionForwardWorkspaceSize(
            handle, input_desc, filter_desc, conv_desc, output_desc,
            CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM, &workspace_bytes
        );

        // 8. Allocate workspace (GPU memory)
        void* d_workspace = nullptr;
        if (cudaMalloc(&d_workspace, workspace_bytes) != cudaSuccess) return -1;

        // 9. Execute convolution
        float alpha = 1.0f;
        float beta = 0.0f;
        cudnnConvolutionForward(
            handle,                               // cuDNN handle
            &alpha,                               // input scale factor, 1.0 means use original input
            input_desc, input_blob.gpu_data,// input descriptor + data
            filter_desc, weight_blob.gpu_data,    // filter descriptor + data (already uploaded to GPU)
            conv_desc, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,  // convolution descriptor + algorithm
            d_workspace,
            workspace_bytes,                      // temporary workspace
            &beta,                                // output scale factor, 0.0 means overwrite
            output_desc,
            output_blob.gpu_data                  // output descriptor + target memory
        );

        // 10. Add bias if bias_term == 1
        if (bias_term == 1)
        {
            cudnnAddTensor(
                handle,
                &alpha,
                bias_desc,
                bias_blob.gpu_data,
                &beta,
                output_desc,
                output_blob.gpu_data
            );
        }

        // 11. Apply activation function if needed
        if (activation_type != 0)
        {
            cudnnActivationMode_t mode;
            switch (activation_type)
            {
            case 1: // ReLU
                mode = CUDNN_ACTIVATION_RELU;
                break;
            case 2: // LeakyReLU -> approximate with ELU
                mode = CUDNN_ACTIVATION_ELU;
                break;
            // case 3: // Clip -> Clipped ReLU
            //     mode = CUDNN_ACTIVATION_CLIPPED_RELU;
            //     relu_clip = 6.0f; // optional upper limit
            //     break;
            case 4: // Sigmoid
                mode = CUDNN_ACTIVATION_SIGMOID;
                break;
            default:
                return -1;
            }
            cudnnSetActivationDescriptor(
                activatio_desc,
                mode,
                CUDNN_PROPAGATE_NAN,
                0.0f
            );
            cudnnActivationForward(
                handle,
                activatio_desc,
                &alpha,
                output_desc,
                output_blob.gpu_data,
                &beta,
                output_desc,
                output_blob.gpu_data
            );
        }

        // 15. Free workspace
        if (d_workspace)
        {
            cudaFree(d_workspace); // Free previously allocated GPU temporary buffer
            d_workspace = nullptr; // Avoid dangling pointer
        }

        // 13. Destroy cuDNN descriptors and handle
        cudnnDestroyTensorDescriptor(input_desc);           // Destroy input tensor descriptor
        cudnnDestroyTensorDescriptor(output_desc);          // Destroy output tensor descriptor
        cudnnDestroyTensorDescriptor(bias_desc);            // Destroy bias descriptor
        cudnnDestroyFilterDescriptor(filter_desc);          // Destroy filter descriptor
        cudnnDestroyConvolutionDescriptor(conv_desc);       // Destroy convolution descriptor
        cudnnDestroyActivationDescriptor(activatio_desc);   // Destroy activation descriptor (if used)
        cudnnDestroy(handle);                               // Destroy cuDNN handle

        return 0;

    }
}
