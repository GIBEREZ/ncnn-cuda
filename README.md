正在挖坑写CUDA实现

底层逻辑修改未实现部分：
test测试系统魔改
ex管理器 input支持Mat类输入，并非CudaMat类
LLM模型支持

| Layer 算子              | FP32 | HALF16 | INT8 | CUDA 实现状态 | 备注                           | CUDA 测试状态 |
| --------------------- | ---- | ------ | ---- |-----------| ---------------------------- | --------- |
| Convolution           | ✅    | ❌      | ❌    | ✅ 已实现     | 卷积                           | ✅ 已测试     |
| ReLU                  | ✅    | ❌      | ❌    | ✅ 已实现     | 激活                           | ✅ 已测试     |
| Reshape               | ✅    | ✅      | ✅    | ✅ 已实现     | 维度变换                         | ✅ 已测试     |
| InnerProduct          | ✅    | ❌      | ❌    | ✅ 已实现     | 全连接                          | ❌ 未测试     |
| AbsVal                | ❌    | ❌      | ❌    | ❌ 未实现     | 绝对值                          | ❌ 未测试     |
| ArgMax                | ❌    | ❌      | ❌    | ❌ 未实现     | 最大值索引/Top-k                  | ❌ 未测试     |
| BatchNorm             | ❌    | ❌      | ❌    | ❌ 未实现     | 批归一化                         | ❌ 未测试     |
| Bias                  | ✅    | ✅      | ✅    | ✅ 已实现     | 加偏置                          | ❌ 未测试     |
| BinaryOp              | ✅    | ❌      | ❌    | ✅ 已实现     | 二元算子                         | ❌ 未测试     |
| BNLL                  | ❌    | ❌      | ❌    | ❌ 未实现     | 双曲对数激活                       | ❌ 未测试     |
| Cast                  | ✅    | ✅      | ✅    | ✅ 已实现     | 类型转换                         | ❌ 未测试     |
| Clip                  | ❌    | ❌      | ❌    | ❌ 未实现     | 限幅                           | ❌ 未测试     |
| Concat                | ❌    | ❌      | ❌    | ❌ 未实现     | 拼接 tensor                    | ❌ 未测试     |
| Convolution1D         | ❌    | ❌      | ❌    | ❌ 未实现     | 一维卷积                         | ❌ 未测试     |
| Convolution3D         | ❌    | ❌      | ❌    | ❌ 未实现     | 三维卷积                         | ❌ 未测试     |
| ConvolutionDepthWise  | ❌    | ❌      | ❌    | ❌ 未实现     | Depthwise 卷积                 | ❌ 未测试     |
| CopyTo                | ❌    | ❌      | ❌    | ❌ 未实现     | 拷贝 tensor                    | ❌ 未测试     |
| Crop                  | ❌    | ❌      | ❌    | ❌ 未实现     | 裁剪                           | ❌ 未测试     |
| CumulativeSum         | ❌    | ❌      | ❌    | ❌ 未实现     | 累加                           | ❌ 未测试     |
| Deconvolution         | ❌    | ❌      | ❌    | ❌ 未实现     | 转置卷积                         | ❌ 未测试     |
| DeformableConv2D      | ❌    | ❌      | ❌    | ❌ 未实现     | 可变形卷积                        | ❌ 未测试     |
| Dequantize            | ❌    | ❌      | ❌    | ❌ 未实现     | 量化反向 (int8→float)            | ❌ 未测试     |
| Diag                  | ❌    | ❌      | ❌    | ❌ 未实现     | 对角操作                         | ❌ 未测试     |
| Dropout               | ❌    | ❌      | ❌    | ❌ 未实现     | Dropout                      | ❌ 未测试     |
| Einsum                | ❌    | ❌      | ❌    | ❌ 未实现     | Einstein 求和                  | ❌ 未测试     |
| Eltwise               | ❌    | ❌      | ❌    | ❌ 未实现     | 元素级运算                        | ❌ 未测试     |
| ELU                   | ❌    | ❌      | ❌    | ❌ 未实现     | ELU 激活                       | ❌ 未测试     |
| Embed                 | ❌    | ❌      | ❌    | ❌ 未实现     | Embedding                    | ❌ 未测试     |
| Erf                   | ❌    | ❌      | ❌    | ❌ 未实现     | Gaussian 错误函数                | ❌ 未测试     |
| ExpandDims            | ❌    | ❌      | ❌    | ❌ 未实现     | 扩展维度                         | ❌ 未测试     |
| Flatten               | ❌    | ❌      | ❌    | ❌ 未实现     | 展平 tensor                    | ❌ 未测试     |
| Flip                  | ❌    | ❌      | ❌    | ❌ 未实现     | 翻转 tensor                    | ❌ 未测试     |
| Fold                  | ❌    | ❌      | ❌    | ❌ 未实现     | 折叠/重构                        | ❌ 未测试     |
| GELU                  | ❌    | ❌      | ❌    | ❌ 未实现     | GELU 激活                      | ❌ 未测试     |
| GLU                   | ❌    | ❌      | ❌    | ❌ 未实现     | Gated Linear Unit            | ❌ 未测试     |
| Gemm                  | ❌    | ❌      | ❌    | ❌ 未实现     | 矩阵乘加                         | ❌ 未测试     |
| GroupNorm             | ❌    | ❌      | ❌    | ❌ 未实现     | 分组归一化                        | ❌ 未测试     |
| GRU                   | ❌    | ❌      | ❌    | ❌ 未实现     | GRU                          | ❌ 未测试     |
| HardSigmoid           | ❌    | ❌      | ❌    | ❌ 未实现     | 硬 Sigmoid                    | ❌ 未测试     |
| HardSwish             | ❌    | ❌      | ❌    | ❌ 未实现     | 硬 Swish                      | ❌ 未测试     |
| InstanceNorm          | ❌    | ❌      | ❌    | ❌ 未实现     | 实例归一化                        | ❌ 未测试     |
| Interp                | ❌    | ❌      | ❌    | ❌ 未实现     | 插值/上采样                       | ❌ 未测试     |
| LayerNorm             | ❌    | ❌      | ❌    | ❌ 未实现     | 层归一化                         | ❌ 未测试     |
| LRN                   | ❌    | ❌      | ❌    | ❌ 未实现     | 局部响应归一化                      | ❌ 未测试     |
| LSTM                  | ❌    | ❌      | ❌    | ❌ 未实现     | LSTM                         | ❌ 未测试     |
| MatMul                | ❌    | ❌      | ❌    | ❌ 未实现     | 矩阵乘法                         | ❌ 未测试     |
| MemoryData            | ❌    | ❌      | ❌    | ❌ 未实现     | 内存数据层                        | ❌ 未测试     |
| Mish                  | ❌    | ❌      | ❌    | ❌ 未实现     | Mish 激活                      | ❌ 未测试     |
| MultiHeadAttention    | ❌    | ❌      | ❌    | ❌ 未实现     | 多头注意力                        | ❌ 未测试     |
| Noop                  | ❌    | ❌      | ❌    | ❌ 未实现     | 无操作                          | ❌ 未测试     |
| Normalize             | ❌    | ❌      | ❌    | ❌ 未实现     | 归一化操作                        | ❌ 未测试     |
| Padding               | ❌    | ❌      | ❌    | ❌ 未实现     | 填充                           | ❌ 未测试     |
| Permute               | ❌    | ❌      | ❌    | ❌ 未实现     | 维度置换                         | ❌ 未测试     |
| PixelShuffle          | ❌    | ❌      | ❌    | ❌ 未实现     | 像素重排                         | ❌ 未测试     |
| Pooling               | ❌    | ❌      | ❌    | ❌ 未实现     | 池化                           | ❌ 未测试     |
| Power                 | ❌    | ❌      | ❌    | ❌ 未实现     | 幂运算                          | ❌ 未测试     |
| PReLU                 | ❌    | ❌      | ❌    | ❌ 未实现     | 参数化 ReLU                     | ❌ 未测试     |
| PriorBox              | ❌    | ❌      | ❌    | ❌ 未实现     | 生成先验框                        | ❌ 未测试     |
| Quantize              | ❌    | ❌      | ❌    | ❌ 未实现     | 量化                           | ❌ 未测试     |
| Reduction             | ❌    | ❌      | ❌    | ❌ 未实现     | 归约                           | ❌ 未测试     |
| Reorg                 | ❌    | ❌      | ❌    | ❌ 未实现     | Reorg                        | ❌ 未测试     |
| Requantize            | ❌    | ❌      | ❌    | ❌ 未实现     | 重新量化                         | ❌ 未测试     |
| RMSNorm               | ❌    | ❌      | ❌    | ❌ 未实现     | RMS 层归一化                     | ❌ 未测试     |
| RNN                   | ❌    | ❌      | ❌    | ❌ 未实现     | RNN                          | ❌ 未测试     |
| ROIPooling            | ❌    | ❌      | ❌    | ❌ 未实现     | ROI Pooling                  | ❌ 未测试     |
| ROIAlign              | ❌    | ❌      | ❌    | ❌ 未实现     | ROI Align                    | ❌ 未测试     |
| Scale                 | ❌    | ❌      | ❌    | ❌ 未实现     | 缩放                           | ❌ 未测试     |
| SDPA                  | ❌    | ❌      | ❌    | ❌ 未实现     | Scaled Dot-Product Attention | ❌ 未测试     |
| SELU                  | ❌    | ❌      | ❌    | ❌ 未实现     | SELU 激活                      | ❌ 未测试     |
| Shrink                | ❌    | ❌      | ❌    | ❌ 未实现     | Shrink                       | ❌ 未测试     |
| ShuffleChannel        | ❌    | ❌      | ❌    | ❌ 未实现     | 通道打散                         | ❌ 未测试     |
| Sigmoid               | ✅    | ❌      | ❌    | ✅ 已实现     | Sigmoid 激活                   | ❌ 未测试     |
| Slice                 | ❌    | ❌      | ❌    | ❌ 未实现     | Slice                        | ❌ 未测试     |
| Softmax               | ❌    | ❌      | ❌    | ❌ 未实现     | Softmax                      | ❌ 未测试     |
| Softplus              | ❌    | ❌      | ❌    | ❌ 未实现     | Softplus                     | ❌ 未测试     |
| Spectrogram           | ❌    | ❌      | ❌    | ❌ 未实现     | 频谱                           | ❌ 未测试     |
| Squeeze               | ❌    | ❌      | ❌    | ❌ 未实现     | 删除维度                         | ❌ 未测试     |
| Swish                 | ❌    | ❌      | ❌    | ❌ 未实现     | Swish                        | ❌ 未测试     |
| TanH                  | ❌    | ❌      | ❌    | ❌ 未实现     | 双曲正切                         | ❌ 未测试     |
| Tile                  | ❌    | ❌      | ❌    | ❌ 未实现     | 重复制                          | ❌ 未测试     |
| Unfold                | ❌    | ❌      | ❌    | ❌ 未实现     | 展开                           | ❌ 未测试     |
| Yolov3DetectionOutput | ❌    | ❌      | ❌    | ❌ 未实现     | YOLOv3 检测输出                  | ❌ 未测试     |
