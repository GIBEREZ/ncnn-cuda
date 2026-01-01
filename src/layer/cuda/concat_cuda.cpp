//
// Created by GIBEREZ on 2026/1/1.
//

#include "concat_cuda.h"

namespace ncnn {
    Concat_cuda::Concat_cuda()
    {
        one_blob_only = false;
        support_inplace = false;
        support_cuda = true;
    }

    int Concat_cuda::load_param(const ParamDict& pd)
    {
        axis = pd.get(0, 0);

        return 0;
    }

    int Concat_cuda::forward(const std::vector<CudaMat>& bottom_blobs, std::vector<CudaMat>& top_blobs, const Option& opt) const
    {
        int dims = bottom_blobs[0].dims;
        int elemsize = bottom_blobs[0].elemsize;
        int positive_axis = axis < 0 ? dims + axis : axis;

        if (dims == 1)
        {
            int top_w = 0;
            for (int i = 0; i < bottom_blobs.size(); i++)
                top_w += bottom_blobs[i].w;

            CudaMat& top_blob = top_blobs[0];
            top_blob.create(top_w, elemsize);
            if (top_blob.empty())
                return -100;

            float* outptr = static_cast<float*>(top_blob.gpu_data);
            for (int i = 0; i < bottom_blobs.size(); i++)
            {
                const CudaMat& bottom_blob = bottom_blobs[i];

                int w = bottom_blob.w;

                const float* ptr = static_cast<const float*>(bottom_blob.gpu_data);
                cudaMemcpy(outptr, ptr, w * elemsize, cudaMemcpyDeviceToDevice);
                outptr += w;
            }
            return 0;
        }

        if (dims == 2 && positive_axis == 0)
        {
            int w = bottom_blobs[0].w;

            int top_h = 0;
            for (size_t b = 0; b < bottom_blobs.size(); b++)
            {
                const CudaMat& bottom_blob = bottom_blobs[b];
                top_h += bottom_blob.h;
            }

            CudaMat& top_blob = top_blobs[0];
            top_blob.create(w, top_h, elemsize);
            if (top_blob.empty())
                return -100;

            float* outptr = static_cast<float*>(top_blob.gpu_data);
            for (int i = 0; i < bottom_blobs.size(); i++)
            {
                const CudaMat& bottom_blob = bottom_blobs[i];

                int size = w * bottom_blob.h;

                const float* ptr = static_cast<const float*>(bottom_blob.gpu_data);
                cudaMemcpy(outptr, ptr, size * elemsize, cudaMemcpyDeviceToDevice);
                outptr += size * elemsize;
            }
            return 0;
        }
        if (dims == 2 && positive_axis == 1)
        {
            int h = bottom_blobs[0].h;

            int top_w = 0;
            for (size_t i = 0; i < bottom_blobs.size(); i++)
            {
                const CudaMat& bottom_blob = bottom_blobs[i];
                top_w += bottom_blob.w;
            }

            CudaMat& top_blob = top_blobs[0];
            top_blob.create(top_w, h, elemsize);
            if (top_blob.empty())
                return -100;

            int offset = 0;
            for (int i = 0; i < bottom_blobs.size(); i++)
            {
                const CudaMat& bottom_blob = bottom_blobs[i];
                Concat_dims2_axis1(bottom_blob, top_blob, h, i, offset, top_w);
                offset += bottom_blob.w;
            }
        }
        if ((dims == 3 || dims == 4) && positive_axis == 0)
        {
            int w = bottom_blobs[0].w;
            int h = bottom_blobs[0].h;
            int d = bottom_blobs[0].d;

            int top_channels = 0;
            for (const auto & bottom_blob : bottom_blobs)
            {
                top_channels += bottom_blob.c;
            }
            CudaMat& top_blob = top_blobs[0];
            top_blob.create(w, h, d, top_channels, elemsize);
            if (top_blob.empty())
                return -100;

            top_blob.dims = dims;
            int q = 0;
            float* outptr_base = static_cast<float*>(top_blob.gpu_data);
            for (size_t i = 0; i < bottom_blobs.size(); i++)
            {
                const CudaMat& bottom_blob = bottom_blobs[i];

                int channels = bottom_blob.c;
                size_t size = bottom_blob.cstep * channels;

                const float* ptr = static_cast<const float*>(bottom_blob.gpu_data);
                float* outptr = outptr_base + q * top_blob.cstep;
                cudaMemcpy(outptr, ptr, size * elemsize, cudaMemcpyDeviceToDevice);

                q += channels;
            }

            return 0;
        }
        if ((dims == 3 && positive_axis == 1) || (dims == 4 && positive_axis == 2))
        {
            int w = bottom_blobs[0].w;
            int d = bottom_blobs[0].d;
            int channels = bottom_blobs[0].c;

            int top_h = 0;
            for (int i = 0; i < bottom_blobs.size(); i++)
            {
                const CudaMat& bottom_blob = bottom_blobs[i];
                top_h += bottom_blob.h;
            }

            CudaMat& top_blob = top_blobs[0];
            top_blob.create(w, top_h, d, channels, elemsize);
            if (top_blob.empty())
                return -100;

            top_blob.dims = dims;

            #pragma omp parallel for num_threads(opt.num_threads)
            float* outptr_base = static_cast<float*>(top_blob.gpu_data);
            for (int q = 0; q < channels; q++)
            {
                float* outptr = outptr_base + q * top_blob.cstep;
                for (int i = 0; i < d; i++)
                {
                    for (int b = 0; b < bottom_blobs.size(); b++)
                    {
                        const CudaMat& bottom_blob = bottom_blobs[b];
                        int size = bottom_blob.w * bottom_blob.h;
                        const float* ptr = static_cast<const float*>(bottom_blob.gpu_data) + q * bottom_blob.cstep + i * bottom_blob.w * bottom_blob.h;
                        cudaMemcpy(outptr, ptr, size * elemsize, cudaMemcpyDeviceToDevice);

                        outptr += size;
                    }
                }
            }

            return 0;
        }
        if ((dims == 3 && positive_axis == 2) || (dims == 4 && positive_axis == 3))
        {
            int h = bottom_blobs[0].h;
            int d = bottom_blobs[0].d;
            int channels = bottom_blobs[0].c;

            // total width
            int top_w = 0;
            for (int i = 0; i < bottom_blobs.size(); i++)
            {
                const CudaMat& bottom_blob = bottom_blobs[i];
                top_w += bottom_blob.w;
            }

            CudaMat& top_blob = top_blobs[0];
            top_blob.create(top_w, h, d, channels, elemsize);
            if (top_blob.empty())
                return -100;

            top_blob.dims = dims;

            float* outptr_base = static_cast<float*>(top_blob.gpu_data);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                float* outptr = outptr_base + q * top_blob.cstep;

                for (int i = 0; i < d; i++)
                {
                    for (int j = 0; j < h; j++)
                    {
                        for (int b = 0; b < bottom_blobs.size(); b++)
                        {
                            const CudaMat& bottom_blob = bottom_blobs[b];

                            const float* ptr =
                                static_cast<const float*>(bottom_blob.gpu_data)
                                + q * bottom_blob.cstep
                                + i * bottom_blob.w * bottom_blob.h
                                + j * bottom_blob.w;

                            cudaMemcpy(outptr, ptr,
                                       bottom_blob.w * elemsize,
                                       cudaMemcpyDeviceToDevice);

                            outptr += bottom_blob.w;
                        }
                    }
                }
            }
            return 0;
        }
        if (dims == 4 && positive_axis == 1)
        {
            int w = bottom_blobs[0].w;
            int h = bottom_blobs[0].h;
            int channels = bottom_blobs[0].c;

            int top_d = 0;
            for (int i = 0; i < bottom_blobs.size(); i++)
            {
                const CudaMat& bottom_blob = bottom_blobs[i];
                top_d += bottom_blob.d;
            }

            CudaMat& top_blob = top_blobs[0];
            top_blob.create(w, h, top_d, channels, elemsize);
            if (top_blob.empty())
                return -100;

            float* outptr_base = static_cast<float*>(top_blob.gpu_data);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                float* outptr = outptr_base + q * top_blob.cstep;

                for (int b = 0; b < bottom_blobs.size(); b++)
                {
                    const CudaMat& bottom_blob = bottom_blobs[b];
                    int size = bottom_blob.w * bottom_blob.h * bottom_blob.d;
                    const float* ptr = static_cast<const float*>(bottom_blob.gpu_data) + q * bottom_blob.cstep;

                    cudaMemcpy(outptr, ptr, size * elemsize, cudaMemcpyDeviceToDevice);
                    outptr += size;
                }
            }
            return 0;
        }
        return 0;
    }


}