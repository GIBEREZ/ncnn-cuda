//
// Created by GIBEREZ on 2025/11/7.
//

#include "reshape_cuda.h"

namespace ncnn {
    Reshape_cuda::Reshape_cuda()
    {
        one_blob_only = false;
        support_cuda = true;
    }

    int Reshape_cuda::load_param(const ParamDict& pd)
    {
        w = pd.get(0, -233);
        h = pd.get(1, -233);
        d = pd.get(11, -233);
        c = pd.get(2, -233);

        ndim = 4;
        if (d == -233)
            ndim = 3;
        if (c == -233)
            ndim = 2;
        if (h == -233)
            ndim = 1;
        if (w == -233)
            ndim = 0;

        shape_expr = pd.get(6, ""); // default empty string

        if (!shape_expr.empty())
        {
            const int blob_count = count_expression_blobs(shape_expr);
            if (blob_count > 1)
                one_blob_only = false;
            std::vector<Mat> blobs(blob_count);
            std::vector<int> outshape;
            int er = eval_list_expression(shape_expr, blobs, outshape);
            if (er != 0)
                return -1;

            ndim = static_cast<int>(outshape.size());
        }

        return 0;
    }

    int Reshape_cuda::forward(const std::vector<CudaMat>& input_blobs, std::vector<CudaMat>& output_blobs, const Option& opt) const
    {
        NCNN_LOGE("  *  Running CUDA Reshape forward");
        const CudaMat& input_blob = input_blobs[0];
        CudaMat& output_blob = output_blobs[0];

        int outw = w;
        int outh = h;
        int outd = d;
        int outc = c;
        int dims = input_blob.dims;
        int total = input_blob.w * input_blob.h * input_blob.d * input_blob.c;

        // Resolve special values: 0 = keep original, -1 = infer from total
        if (outw == 0)   outw = dims >= 1 ? input_blob.w : outw;
        if (outh == 0)   outh = dims >= 2 ? input_blob.h : outh;
        if (outd == 0)   outd = dims >= 4 ? input_blob.d : outd;
        if (outc == 0)   outc = dims >= 3 ? input_blob.c : outc;
        if (outw == -1)  outw = total / (std::max(outh, 1) * std::max(outd, 1) * std::max(outc, 1));
        if (outh == -1)  outh = total / (std::max(outw, 1) * std::max(outd, 1) * std::max(outc, 1));
        if (outd == -1)  outd = total / (std::max(outw, 1) * std::max(outh, 1) * std::max(outc, 1));
        if (outc == -1)  outc = total / (std::max(outw, 1) * std::max(outh, 1) * std::max(outd, 1));

        NCNN_LOGE("  *  reshape %d %d %d %d -> %d %d %d %d  ndim=%d",
                  input_blob.w, input_blob.h, input_blob.d, input_blob.c,
                  outw, outh, outd, outc, ndim);

        // Reshape is metadata-only: create a view with new dims, same GPU data
        if (ndim == 1)
            output_blob = input_blob.reshape(outw);
        else if (ndim == 2)
            output_blob = input_blob.reshape(outw, outh);
        else if (ndim == 3)
            output_blob = input_blob.reshape(outw, outh, outc);
        else if (ndim == 4)
            output_blob = input_blob.reshape(outw, outh, outd, outc);
        else
            output_blob = input_blob; // no-op

        NCNN_LOGE("  *  forward output_blob w=%d,h=%d,d=%d,c=%d,dims=%d",output_blob.w,output_blob.h,output_blob.d,output_blob.c,output_blob.dims);
        if (output_blob.empty() || output_blob.gpu_data == nullptr) NCNN_LOGE("  *  output blob gpu_data == nullptr");
        NCNN_LOGE("  *  CUDA Reshape forward done");

        return 0;
    }


}
