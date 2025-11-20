//
// Created by GIBEREZ on 2025/11/21.
//

#include "cast_cuda.h"


namespace ncnn {
    Cast_cuda::Cast_cuda()
    {
        one_blob_only = true;
        support_cuda = true;
    }

    int Cast_cuda::load_param(const ParamDict& pd)
    {
        type_from = pd.get(0, 0);
        type_to = pd.get(1, 0);

        if (type_from == 4) type_from = 2;
        if (type_to == 4) type_to = 2;

        return 0;
    }

    int Cast_cuda::forward(const CudaMat& bottom_blob, CudaMat& top_blob, const Option& opt) const
    {

        return 0;
    }

} // namespace ncnn