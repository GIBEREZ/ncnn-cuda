//
// Created by GIBEREZ on 2025/10/20.
//

#if NCNN_CUDA
    #include <cstdio>
    #include <cmath>
    #include "net.h"
    #include "command.h"

    int main() {
        ncnn::get_device_properties();

        // 构造输入 Mat: 3 通道 32x32
        ncnn::Mat input(32, 32, 3);
        input.fill(0.01f);

        // ==============================
        // CUDA 推理
        // ==============================
        ncnn::Net net_cuda;
        net_cuda.opt.use_cuda = true;

        int ret = net_cuda.load_param("D:/software/Projects/PythonProjects/torch/model.ncnn.param");
        if (ret != 0) { fprintf(stderr, "CUDA: Failed to load param\n"); return -1; }
        ret = net_cuda.load_model("D:/software/Projects/PythonProjects/torch/model.ncnn.bin");
        if (ret != 0) { fprintf(stderr, "CUDA: Failed to load model\n"); return -1; }

        ncnn::Mat output_cuda;
        {
            ncnn::Extractor ex = net_cuda.create_extractor();
            ex.input("in0", input);
            ex.extract("out0", output_cuda);
        }

        NCNN_LOGE("=== CUDA output: total=%zu dims=%d w=%d ===",
                  output_cuda.total(), output_cuda.dims, output_cuda.w);

        // ==============================
        // CPU 参考推理
        // ==============================
        ncnn::Net net_cpu;
        net_cpu.opt.use_cuda = false;

        ret = net_cpu.load_param("D:/software/Projects/PythonProjects/torch/model.ncnn.param");
        if (ret != 0) { fprintf(stderr, "CPU: Failed to load param\n"); return -1; }
        ret = net_cpu.load_model("D:/software/Projects/PythonProjects/torch/model.ncnn.bin");
        if (ret != 0) { fprintf(stderr, "CPU: Failed to load model\n"); return -1; }

        ncnn::Mat output_cpu;
        {
            ncnn::Extractor ex = net_cpu.create_extractor();
            ex.input("in0", input);
            ex.extract("out0", output_cpu);
        }

        NCNN_LOGE("=== CPU  output: total=%zu dims=%d w=%d ===",
                  output_cpu.total(), output_cpu.dims, output_cpu.w);

        // ==============================
        // 对比
        // ==============================
        NCNN_LOGE("--- i  |  CUDA        CPU         diff");
        bool all_match = true;
        int count = output_cuda.w;
        if (output_cpu.dims == output_cuda.dims && output_cpu.w == output_cuda.w)
        {
            const float* p_cuda = (const float*)output_cuda.data;
            const float* p_cpu  = (const float*)output_cpu.data;
            for (int i = 0; i < count; i++)
            {
                float diff = fabsf(p_cuda[i] - p_cpu[i]);
                if (diff > 0.001f) all_match = false;
                NCNN_LOGE("  %3d  |  %+.6f  %+.6f  %s%.6f",
                          i, p_cuda[i], p_cpu[i],
                          diff > 0.001f ? "*** " : "    ", diff);
            }
        }
        else
        {
            NCNN_LOGE("  dimension mismatch! CUDA dims=%d w=%d  CPU dims=%d w=%d",
                      output_cuda.dims, output_cuda.w, output_cpu.dims, output_cpu.w);
            all_match = false;
        }

        NCNN_LOGE("=== %s ===", all_match ? "ALL MATCH" : "MISMATCH DETECTED");
        return 0;
    }
#endif