// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef NCNN_COMMAND_H
#define NCNN_COMMAND_H

#include "platform.h"

#if NCNN_CUDA

#include "mat.h"

namespace ncnn {
    // 该类不需要添加NCNN_EXPORT宏定义，因为CUDA库已被单独编译成lib
    class CudaCompute
    {
    public:
        // 防止编译器自动调用构造函数进行“类型转换”，要求必须“显式调用”。
        explicit CudaCompute();
        virtual ~CudaCompute();
    public:
        void Upload_Device(const Mat& src, CudaMat& dst, const Option& option);    // 将CPU端Mat数据上传到GPU设备端CudaMat
        void Download_Device(const CudaMat& src, Mat& dst, const Option& option);  // 将GPU设备端CudaMat数据下载回CPU端Mat
        void Clone_Device(const Mat& src, CudaMat& dst, const Option& option);     // 从CPU端Mat克隆生成GPU端CudaMat
        void Clone_Device(const CudaMat& src, Mat& dst, const Option& option);     // 从GPU端CudaMat克隆生成CPU端Mat
    };
}
#endif

#if NCNN_VULKAN

#include "mat.h"

namespace ncnn {

class Pipeline;
#if NCNN_PLATFORM_API
#if __ANDROID_API__ >= 26
class ImportAndroidHardwareBufferPipeline;
#endif // __ANDROID_API__ >= 26
#endif // NCNN_PLATFORM_API
class VkComputePrivate;
class NCNN_EXPORT VkCompute
{
public:
    explicit VkCompute(const VulkanDevice* vkdev);
    virtual ~VkCompute();

public:
    void record_upload(const Mat& src, VkMat& dst, const Option& opt);

    void record_download(const VkMat& src, Mat& dst, const Option& opt);

    void record_clone(const Mat& src, VkMat& dst, const Option& opt);

    void record_clone(const Mat& src, VkImageMat& dst, const Option& opt);

    void record_clone(const VkMat& src, Mat& dst, const Option& opt);

    void record_clone(const VkImageMat& src, Mat& dst, const Option& opt);

    void record_clone(const VkMat& src, VkMat& dst, const Option& opt);

    void record_clone(const VkImageMat& src, VkImageMat& dst, const Option& opt);

    void record_clone(const VkMat& src, VkImageMat& dst, const Option& opt);

    void record_clone(const VkImageMat& src, VkMat& dst, const Option& opt);

    void record_pipeline(const Pipeline* pipeline, const std::vector<VkMat>& bindings, const std::vector<vk_constant_type>& constants, const VkMat& dispatcher);

    void record_pipeline(const Pipeline* pipeline, const std::vector<VkImageMat>& bindings, const std::vector<vk_constant_type>& constants, const VkImageMat& dispatcher);

    void record_pipeline(const Pipeline* pipeline, const std::vector<VkMat>& buffer_bindings, const std::vector<VkImageMat>& image_bindings, const std::vector<vk_constant_type>& constants, const VkMat& dispatcher);
    void record_pipeline(const Pipeline* pipeline, const std::vector<VkMat>& buffer_bindings, const std::vector<VkImageMat>& image_bindings, const std::vector<vk_constant_type>& constants, const VkImageMat& dispatcher);
    void record_pipeline(const Pipeline* pipeline, const std::vector<VkMat>& buffer_bindings, const std::vector<VkImageMat>& image_bindings, const std::vector<vk_constant_type>& constants, const Mat& dispatcher);

#if NCNN_BENCHMARK
    void record_write_timestamp(uint32_t query);
#endif // NCNN_BENCHMARK

#if NCNN_PLATFORM_API
#if __ANDROID_API__ >= 26
    void record_import_android_hardware_buffer(const ImportAndroidHardwareBufferPipeline* pipeline, const VkImageMat& src, const VkMat& dst);
#endif // __ANDROID_API__ >= 26
#endif // NCNN_PLATFORM_API

    int submit_and_wait();

    int reset();

#if NCNN_BENCHMARK
    int create_query_pool(uint32_t query_count);

    int get_query_pool_results(uint32_t first_query, uint32_t query_count, std::vector<uint64_t>& results);
#endif // NCNN_BENCHMARK

protected:
    const VulkanDevice* vkdev;

    void barrier_readwrite(const VkMat& binding);
    void barrier_readwrite(const VkImageMat& binding);
    void barrier_readonly(const VkImageMat& binding);

private:
    VkComputePrivate* const d;
};

class VkTransferPrivate;
class NCNN_EXPORT VkTransfer
{
public:
    explicit VkTransfer(const VulkanDevice* vkdev);
    virtual ~VkTransfer();

public:
    void record_upload(const Mat& src, VkMat& dst, const Option& opt, bool flatten = true);

    int submit_and_wait();

protected:
    const VulkanDevice* vkdev;

private:
    VkTransferPrivate* const d;
};

} // namespace ncnn

#endif // NCNN_VULKAN

#endif // NCNN_COMMAND_H
