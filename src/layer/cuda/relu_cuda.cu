#include "relu_cuda.h"

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
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < number) output_blob[idx] = input_blob[idx] > 0.0f ? input_blob[idx] : 0.0f;
}

extern "C" void relu_cuda(const float* input_blob, float* output_blob, int number)
{
    int threadsPerBlock = 1024;
    int blocksPerGrid = (number + threadsPerBlock - 1) / threadsPerBlock;
    relu_kernel_cuda<<<blocksPerGrid, threadsPerBlock>>>(input_blob, output_blob, number);
    cudaDeviceSynchronize();
}
