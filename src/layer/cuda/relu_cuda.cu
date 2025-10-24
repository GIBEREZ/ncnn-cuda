#include "relu_cuda.h"
#include <cuda_runtime.h>

namespace ncnn {
    /**
     * CUDA�汾�����kernel() �� CUDA�汾�����kernel��CPP API�ӿ�
     * @param input_blob ��������-��������ָ��
     * @param output_blob �������-��������ָ��
     * @param number ��������Ԫ�ظ���
     * ��CUDA kernel�x(Ҳ���� const float* x)ֻ��һ��ָ�������ڴ����������ָ�룬������֪��Ҳ������ά�ȣ�shape����
     */
    __global__ void relu_kernel_cuda(const float* input_blob, float* output_blob, int number)
    {
        // ����ȫ���߳�������global thread index��
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x * 4;
        unsigned int idxElement = idx * 4;
        if (idxElement + 3 < number) {
            float4 vec_in = *(float4*)&input_blob[idxElement];
            float4 vec_out;

            vec_out.x = vec_in.x > 0.0f ? vec_in.x : 0.0f;
            vec_out.y = vec_in.y > 0.0f ? vec_in.y : 0.0f;
            vec_out.z = vec_in.z > 0.0f ? vec_in.z : 0.0f;
            vec_out.w = vec_in.w > 0.0f ? vec_in.w : 0.0f;

            *(float4*)&output_blob[idxElement] = vec_out;
        }
        else {
            for (int i = 0; i < 4; i++) {
                unsigned int elem_idx = idxElement + i;
                if (elem_idx < number) {
                    output_blob[elem_idx] = input_blob[elem_idx] > 0.0f ? input_blob[elem_idx] : 0.0f;
                }
            }
        }
    }
   void relu_cuda(const float* input_blob, float* output_blob, int number)
    {
        // ����ÿ���߳̿���߳�����
        int threadsPerBlock = 1024;
        // �����ܹ���Ҫ���ٸ��߳��������������顣��Ϊÿ���̴߳���4��Ԫ�أ��������߳���Ӧ����Ԫ����������4��������ȡ��������Ϊfloat��4�ֽڣ�16�ֽ�/4�ֽ�=4��
        int totalThreadsNeeded = (number + 4 - 1) / 4;
        // �����������߳̿��������ÿ���߳̿���256���̣߳��������߳�������ÿ���߳̿���߳�����������ȡ����
        int blocksPerGrid = (totalThreadsNeeded + threadsPerBlock - 1) / threadsPerBlock;
        relu_kernel_cuda<<<blocksPerGrid, threadsPerBlock>>>(input_blob, output_blob, number);
        // ͬ���豸���ȴ��ں�ִ����ɡ��������������������CPU��ֱ���豸��GPU���ϵ����в�����ɡ�
        cudaDeviceSynchronize();
    }
}