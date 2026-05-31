//
// Created by GIBEREZ on 2025/12/26.
//

#ifndef NCNN_GEMM_CUDA_H
#define NCNN_GEMM_CUDA_H
#include "layer.h"

namespace ncnn {
    class Gemm_cuda : public Layer
    {
    public:
        Gemm_cuda();
        int load_param(const ParamDict& pd) override;
        int load_model(const ModelBin& mb) override;

        using Layer::forward;
        int forward(const CudaMat& input_blob, CudaMat& output_blob, const Option& opt) const override;
        int forward(const std::vector<CudaMat>& bottom_blobs, std::vector<CudaMat>& top_blobs, const Option& opt) const override;
        int gemm_cuda(const std::vector<CudaMat>& bottom_blobs, std::vector<CudaMat>& top_blobs, const Option& opt) const;
    public:
        float alpha;
        float beta;
        // transpose
        int transA;
        int transB;

        // Is it a constant matrix
        int constantA;
        int constantB;
        int constantC;
        CudaMat model_A;
        CudaMat model_B;
        CudaMat model_C;

        // constant matrix shape
        int constantM;
        int constantN;
        int constantK;
        int constant_TILE_M;
        int constant_TILE_N;
        int constant_TILE_K;

        // constant matrix broadcasting strategy
        int constant_broadcast_type_C;

        // Output layout and memory format
        int output_N1M;
        int output_elemtype; // 0=auto 1=fp32
        int output_transpose;
    };
}

#endif //NCNN_GEMM_CUDA_H
