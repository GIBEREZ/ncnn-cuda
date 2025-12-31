//
// Created by GIBEREZ on 2025/12/26.
//

#include "gemm_cuda.h"

namespace ncnn {
    Gemm_cuda::Gemm_cuda()
    {
        one_blob_only = false;
        support_inplace = false;
        support_cuda = true;
    }

    int Gemm_cuda::load_param(const ParamDict& pd)
    {
        alpha = pd.get(0, 1.f);
        beta = pd.get(1, 1.f);
        transA = pd.get(2, 0);
        transB = pd.get(3, 0);
        constantA = pd.get(4, 0);
        constantB = pd.get(5, 0);
        constantC = pd.get(6, 0);
        constantM = pd.get(7, 0);
        constantN = pd.get(8, 0);
        constantK = pd.get(9, 0);
        constant_broadcast_type_C = pd.get(10, 0);
        output_N1M = pd.get(11, 0);
        output_elemtype = pd.get(13, 0);
        output_transpose = pd.get(14, 0);
        constant_TILE_M = pd.get(20, 0);
        constant_TILE_N = pd.get(21, 0);
        constant_TILE_K = pd.get(22, 0);

        if (constantA == 1 && (constantM == 0 || constantK == 0))
        {
            NCNN_LOGE("constantM and constantK must be non-zero when constantA enabled");
            return -1;
        }

        if (constantB == 1 && (constantN == 0 || constantK == 0))
        {
            NCNN_LOGE("constantN and constantK must be non-zero when constantB enabled");
            return -1;
        }

        if (constantC == 1 && (constant_broadcast_type_C < -1 || constant_broadcast_type_C > 4))
        {
            NCNN_LOGE("constant_broadcast_type_C must be -1 or 0~4 when constantC enabled");
            return -1;
        }

        if (constantA == 0 && constantB == 1 && constantC == 1)
            one_blob_only = true;

        if (constantA == 1 && constantB == 0 && constantC == 1)
            one_blob_only = true;

        if (constantA == 1 && constantB == 1 && constantC == 0)
            one_blob_only = true;

        return 0;
    }

    int Gemm_cuda::load_model(const ModelBin& mb)
    {
        if (constantA == 1)
        {
            if (transA == 0)
                model_A = CudaMat(mb.load(constantK, constantM, 0));
            else
                model_A = CudaMat(mb.load(constantM, constantK, 0));
            if (model_A.empty())
                return -100;
        }

        if (constantB == 1)
        {
            if (transB == 0)
                model_B = CudaMat(mb.load(constantN, constantK, 0));
            else
                model_B = CudaMat(mb.load(constantK, constantN, 0));
            if (model_B.empty())
                return -100;
        }

        if (constantC == 1 && constant_broadcast_type_C != -1)
        {
            if (constant_broadcast_type_C == 0)
                model_C = CudaMat(mb.load(1, 0));
            if (constant_broadcast_type_C == 1)
                model_C = CudaMat(mb.load(constantM, 0));
            if (constant_broadcast_type_C == 2)
                model_C = CudaMat(mb.load(1, constantM, 0));
            if (constant_broadcast_type_C == 3)
                model_C = CudaMat(mb.load(constantN, constantM, 0));
            if (constant_broadcast_type_C == 4)
                model_C = CudaMat(mb.load(constantN, 1, 0));
            if (model_C.empty())
                return -100;
        }
        return 0;
    }

    int Gemm_cuda::forward(const CudaMat& input_blob, CudaMat& output_blob, const Option& opt) const
    {
        std::vector bottom_blobs(1, input_blob);
        std::vector top_blobs(1, output_blob);
        int ret = forward(bottom_blobs, top_blobs, opt);
        output_blob = top_blobs[0];
        return ret;
    }

    int Gemm_cuda::forward(const std::vector<CudaMat>& bottom_blobs, std::vector<CudaMat>& top_blobs, const Option& opt) const
    {
        return 0;
    }


}