//
// Created by GIBEREZ on 2025/11/4.
//
#include "convolution_cuda.h"
#include <cuda_runtime.h>
#include <cudnn.h>

namespace ncnn {
    int Convolution_cuda::relu_cuda(const CudaMat& input_blob, CudaMat& output_blob, const Option& opt)
    {
        // 1. 创建 cuDNN 句柄
        cudnnHandle_t handle;
        cudnnCreate(&handle);

        // 2. 创建张量描述符（输入、权重、输出）
        cudnnTensorDescriptor_t input_desc, output_desc, bias_desc;
        cudnnFilterDescriptor_t filter_desc;
        cudnnConvolutionDescriptor_t conv_desc;

        // 3. 初始化这些描述符
        cudnnCreateTensorDescriptor(&input_desc);                   // 创建输入张量描述符
        cudnnCreateTensorDescriptor(&output_desc);                  // 创建输出张量描述符
        cudnnCreateTensorDescriptor(&bias_desc);                    // 创建偏置张量描述符
        cudnnCreateFilterDescriptor(&filter_desc);                  // 创建卷积核描述符
        cudnnCreateConvolutionDescriptor(&conv_desc);               // 创建卷积操作描述符

        // 4.配置输入shape尺寸描述符
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

        if (input_blob.dims == 1)
        {
            cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW,
                                       cudnn_dtype, 1, 1, 1, input_blob.w);
        }
        else if (input_blob.dims == 2)
        {
            cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW,
                                       cudnn_dtype, 1, 1, input_blob.h, input_blob.w);
        }
        else if (input_blob.dims == 3)
        {
            cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW,
                                       cudnn_dtype, 1, input_blob.c, input_blob.h, input_blob.w);
        }
        else if (input_blob.dims == 4)
        {
            cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW,
                                       cudnn_dtype, input_blob.d, input_blob.c, input_blob.h, input_blob.w);
        }
        else
        {
            return -1;
        }

        // 5.配置卷积核描述符
        cudnnSetFilter4dDescriptor(filter_desc, cudnn_dtype,
                                   CUDNN_TENSOR_NCHW, num_output, weight_data_size / (num_output * kernel_h * kernel_w), kernel_h, kernel_w);

        // 6.配置卷积参数（padding、stride、dilation）,cuDNN本身不支持非对称padding
        cudnnSetConvolution2dDescriptor(conv_desc,
                                        std::max(pad_top, pad_bottom), std::max(pad_left, pad_right),
                                        stride_h, stride_w,
                                        dilation_h, dilation_w,
                                        CUDNN_CROSS_CORRELATION,
                                        cudnn_dtype);

        // 7.配置输出shape尺寸描述符
        int out_h = (input_blob.h + pad_top + pad_bottom - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
        int out_w = (input_blob.w + pad_left + pad_right - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
        cudnnSetTensor4dDescriptor(output_desc,
            CUDNN_TENSOR_NCHW, cudnn_dtype,
            input_blob.d,
            num_output,
            out_h, out_w);

        // 8.选择卷积算法
        cudnnConvolutionFwdAlgo_t algo;
        cudnnConvolutionFwdAlgoPerf_t perfResults;
        int returnedAlgoCount;
        cudnnGetConvolutionForwardAlgorithm_v7(
            handle, input_desc, filter_desc, conv_desc, output_desc,
            1, &returnedAlgoCount, &perfResults);
        algo = perfResults.algo;

        // 9.计算工作空间大小
        size_t workspace_bytes = 0;
        cudnnGetConvolutionForwardWorkspaceSize(
            handle, input_desc, filter_desc, conv_desc, output_desc,
            algo, &workspace_bytes);

        // 10.分配工作空间（GPU内存）
        void* d_workspace = nullptr;
        cudaMalloc(&d_workspace, workspace_bytes);

        // 11.分配输出张量显存
        output_blob.cstep = alignSize(num_output, 16) * out_h * out_w;
        output_blob.alloc_bytes = output_blob.d * output_blob.cstep * output_blob.elemsize;
        cudaMalloc(&output_blob.data, output_blob.alloc_bytes);

        // 12.执行卷积
        // cudnnConvolutionForward(
        //     handle,
        //     &alpha,                             // 输入缩放系数
        //     input_desc, input_blob.data,  // 输入描述符+数据
        //     filter_desc, weight_data,        // 卷积核描述符+数据（需提前上传GPU）
        //     conv_desc, algo,                    // 卷积描述符+算法
        //     d_workspace,
        //     workspace_bytes,                    // 临时工作空间
        //     &beta,                              // 输出缩放系数
        //     output_desc,
        //     output_blob.data
        //     );// 输出描述符+目标内存
        //
        // // 13.如果有偏置项，则加上偏置
        // if (bias_term == 1)
        // {
        //     cudnnAddTensor(
        //         handle,
        //
        //         );
        // }

        // 14.释放工作空间

        // 15.销毁cuDNN描述符与句柄
        return 0;
    }
}
