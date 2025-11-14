//
// Created by GIBEREZ on 2025/11/4.
//
#include "convolution_cuda.h"
#include <cuda_runtime.h>
#include <cudnn.h>

namespace ncnn {
    int Convolution_cuda::Convolution_cuda_forward(const CudaMat& input_blob, CudaMat& output_blob, const Option& opt) const
    {
        // 1. 创建 cuDNN 句柄
        cudnnHandle_t handle;
        if (cudnnCreate(&handle) != CUDNN_STATUS_SUCCESS) return -1;

        // 2. 创建张量描述符（输入、权重、输出）
        cudnnTensorDescriptor_t input_desc, output_desc, bias_desc;
        cudnnFilterDescriptor_t filter_desc;
        cudnnConvolutionDescriptor_t conv_desc;
        cudnnActivationDescriptor_t activatio_desc;

        // 3. 初始化这些描述符
        cudnnCreateTensorDescriptor(&input_desc);                   // 创建输入张量描述符
        cudnnCreateTensorDescriptor(&output_desc);                  // 创建输出张量描述符
        cudnnCreateFilterDescriptor(&filter_desc);                  // 创建卷积核描述符
        cudnnCreateConvolutionDescriptor(&conv_desc);               // 创建卷积操作描述符
        cudnnCreateTensorDescriptor(&bias_desc);                    // 创建偏置张量描述符
        cudnnCreateActivationDescriptor(&activatio_desc);           // 创建激活函数操作描述符

        // 4.配置输入输出shape尺寸描述符
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

        // 5.配置卷积核描述符
        cudnnSetFilter4dDescriptor(filter_desc, cudnn_dtype,
                                   CUDNN_TENSOR_NCHW,
                                   num_output,
                                   weight_data_size / (num_output * kernel_h * kernel_w),
                                   kernel_h,
                                   kernel_w
        );

        // 6.配置卷积参数（padding、stride、dilation）,cuDNN本身不支持非对称padding
        cudnnSetConvolution2dDescriptor(conv_desc,
                                        std::max(pad_top, pad_bottom), std::max(pad_left, pad_right),
                                        stride_h, stride_w,
                                        dilation_h, dilation_w,
                                        CUDNN_CROSS_CORRELATION,
                                        cudnn_dtype
        );

        // 7.计算工作空间大小
        size_t workspace_bytes = 0;
        cudnnGetConvolutionForwardWorkspaceSize(
            handle, input_desc, filter_desc, conv_desc, output_desc,
            CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM, &workspace_bytes
        );

        // 8.分配工作空间（GPU内存）
        void* d_workspace = nullptr;
        if (cudaMalloc(&d_workspace, workspace_bytes) != cudaSuccess) return -1;

        // 9.分配输出张量显存
        output_blob.cstep = alignSize(num_output, 16) * out_h * out_w;
        output_blob.alloc_bytes = output_blob.d * output_blob.cstep * output_blob.elemsize;
        cudaMalloc(&output_blob.data, output_blob.alloc_bytes);

        // 10.执行卷积
        float alpha = 1.0f;
        float beta = 0.0f;
        cudnnConvolutionForward(
            handle,                               // cudnn句柄
            &alpha,                               // 输入缩放系数，1.0f表示使用原输入
            input_desc, input_blob.data,    // 输入描述符+数据
            filter_desc, weight_blob,          // 卷积核描述符+数据（需提前上传GPU）
            conv_desc, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,  // 卷积描述符+算法
            d_workspace,
            workspace_bytes,                      // 临时工作空间
            &beta,                                // 输出缩放系数，0.0f表示直接覆盖原输入
            output_desc,
            output_blob.data
        );// 输出描述符+目标内存

        // 11.如果有偏置项，则加上偏置
        if (bias_term == 1)
        {
            cudnnAddTensor(
                handle,
                &alpha,
                bias_desc,
                bias_blob.data,
                &beta,
                output_desc,
                output_blob.data
            );
        }

        // 12.如果需要激活函数，则进行激活函数
        if (activation_type != 0)
        {
            cudnnActivationMode_t mode;
            switch (activation_type)
            {
            case 1: // ReLU
                mode = CUDNN_ACTIVATION_RELU;
                break;
            case 2: // LeakyReLU -> 用 ELU 近似
                mode = CUDNN_ACTIVATION_ELU;
                break;
            // case 3: // Clip -> Clipped ReLU
            //     mode = CUDNN_ACTIVATION_CLIPPED_RELU;
            //     relu_clip = 6.0f; // 可以自定义上限，例如 6
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
                output_blob.data,
                &beta,
                output_desc,
                output_blob.data
            );
        }

        // 13.释放工作空间
        if (d_workspace)
        {
            cudaFree(d_workspace); // 释放之前用 cudaMalloc 分配的 GPU 临时缓冲区
            d_workspace = nullptr; // 避免悬空指针
        }

        // 14.销毁cuDNN描述符与句柄
        cudnnDestroyTensorDescriptor(input_desc);           // 销毁输入张量描述符
        cudnnDestroyTensorDescriptor(output_desc);          // 销毁输出张量描述符
        cudnnDestroyTensorDescriptor(bias_desc);            // 销毁 bias 张量描述符
        cudnnDestroyFilterDescriptor(filter_desc);          // 销毁卷积核描述符
        cudnnDestroyConvolutionDescriptor(conv_desc);       // 销毁卷积描述符
        cudnnDestroyActivationDescriptor(activatio_desc);   // 销毁激活描述符（如果使用了）
        cudnnDestroy(handle);                               // 销毁 cuDNN 上下文句柄

        return 0;
    }
}
