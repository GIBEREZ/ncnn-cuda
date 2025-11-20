正在挖坑写CUDA实现

底层逻辑修改未实现部分：
test测试系统魔改
ex管理器 input支持Mat类输入，并非CudaMat类
LLM模型支持

| Layer 算子             | FP32 精度 | FP16 精度 | CUDA 实现状态 | 备注 |
|----------------------|-----------|-----------|---------------|------|
| Convolution          | ✅        | ❌        | ✅ 已实现     | 卷积 |
| ReLU                 | ✅        | ❌        | ✅ 已实现     | 激活 |
| Reshape              | ✅        | ✅        | ✅ 已实现     | 维度变换 |
| InnerProduct         | ✅        | ❌        | ✅ 已实现     | 全连接 |
| AbsVal               | ❌        | ❌        | ❌ 未实现     | 绝对值 |
| ArgMax               | ❌        | ❌        | ❌ 未实现     | 最大值索引/Top-k |
| BatchNorm            | ❌        | ❌        | ❌ 未实现     | 批归一化 |
| Bias                 | ❌        | ❌        | ❌ 未实现     | 加偏置 |
| BinaryOp             | ❌        | ❌        | ❌ 未实现     | 二元算子 |
| BNLL                 | ❌        | ❌        | ❌ 未实现     | 双曲对数激活 |
| Cast                 | ❌        | ❌        | ❌ 未实现     | 类型转换 |
| Clip                 | ❌        | ❌        | ❌ 未实现     | 限幅 |
| Concat               | ❌        | ❌        | ❌ 未实现     | 拼接 tensor |
| Convolution1D        | ❌        | ❌        | ❌ 未实现     | 一维卷积 |
| Convolution3D        | ❌        | ❌        | ❌ 未实现     | 三维卷积 |
| ConvolutionDepthWise | ❌        | ❌        | ❌ 未实现     | Depthwise 卷积 |
| CopyTo               | ❌        | ❌        | ❌ 未实现     | 拷贝 tensor |
| Crop                 | ❌        | ❌        | ❌ 未实现     | 裁剪 |
| CumulativeSum        | ❌        | ❌        | ❌ 未实现     | 累加 |
| Deconvolution        | ❌        | ❌        | ❌ 未实现     | 转置卷积 |
| DeformableConv2D     | ❌        | ❌        | ❌ 未实现     | 可变形卷积 |
| Dequantize           | ❌        | ❌        | ❌ 未实现     | 量化反向 (int8→float) |
| Diag                 | ❌        | ❌        | ❌ 未实现     | 对角操作 |
| Dropout              | ❌        | ❌        | ❌ 未实现     | Dropout |
| Einsum               | ❌        | ❌        | ❌ 未实现     | Einstein 求和 |
| Eltwise              | ❌        | ❌        | ❌ 未实现     | 元素级运算 |
| ELU                  | ❌        | ❌        | ❌ 未实现     | ELU 激活 |
| Embed                | ❌        | ❌        | ❌ 未实现     | Embedding |
| Erf                  | ❌        | ❌        | ❌ 未实现     | Gaussian 错误函数 |
| ExpandDims           | ❌        | ❌        | ❌ 未实现     | 扩展维度 |
| Flatten              | ❌        | ❌        | ❌ 未实现     | 展平 tensor |
| Flip                 | ❌        | ❌        | ❌ 未实现     | 翻转 tensor |
| Fold                 | ❌        | ❌        | ❌ 未实现     | 折叠/重构 |
| GELU                 | ❌        | ❌        | ❌ 未实现     | GELU 激活 |
| GLU                  | ❌        | ❌        | ❌ 未实现     | Gated Linear Unit |
| Gemm                 | ❌        | ❌        | ❌ 未实现     | 矩阵乘加 |
| GroupNorm            | ❌        | ❌        | ❌ 未实现     | 分组归一化 |
| GRU                  | ❌        | ❌        | ❌ 未实现     | GRU |
| HardSigmoid          | ❌        | ❌        | ❌ 未实现     | 硬 Sigmoid |
| HardSwish            | ❌        | ❌        | ❌ 未实现     | 硬 Swish |
| InstanceNorm         | ❌        | ❌        | ❌ 未实现     | 实例归一化 |
| Interp               | ❌        | ❌        | ❌ 未实现     | 插值/上采样 |
| LayerNorm            | ❌        | ❌        | ❌ 未实现     | 层归一化 |
| LRN                  | ❌        | ❌        | ❌ 未实现     | 局部响应归一化 |
| LSTM                 | ❌        | ❌        | ❌ 未实现     | LSTM |
| MatMul               | ❌        | ❌        | ❌ 未实现     | 矩阵乘法 |
| MemoryData           | ❌        | ❌        | ❌ 未实现     | 内存数据层 |
| Mish                 | ❌        | ❌        | ❌ 未实现     | Mish 激活 |
| MultiHeadAttention   | ❌        | ❌        | ❌ 未实现     | 多头注意力 |
| Noop                 | ❌        | ❌        | ❌ 未实现     | 无操作 |
| Normalize            | ❌        | ❌        | ❌ 未实现     | 归一化操作 |
| Padding              | ❌        | ❌        | ❌ 未实现     | 填充 |
| Permute              | ❌        | ❌        | ❌ 未实现     | 维度置换 |
| PixelShuffle         | ❌        | ❌        | ❌ 未实现     | 像素重排 |
| Pooling              | ❌        | ❌        | ❌ 未实现     | 池化 |
| Power                | ❌        | ❌        | ❌ 未实现     | 幂运算 |
| PReLU                | ❌        | ❌        | ❌ 未实现     | 参数化 ReLU |
| PriorBox             | ❌        | ❌        | ❌ 未实现     | 生成先验框 |
| Quantize             | ❌        | ❌        | ❌ 未实现     | 量化 |
| Reduction            | ❌        | ❌        | ❌ 未实现     | 归约 |
| Reorg                | ❌        | ❌        | ❌ 未实现     | Reorg |
| Requantize           | ❌        | ❌        | ❌ 未实现     | 重新量化 |
| RMSNorm              | ❌        | ❌        | ❌ 未实现     | RMS 层归一化 |
| RNN                  | ❌        | ❌        | ❌ 未实现     | RNN |
| ROIPooling           | ❌        | ❌        | ❌ 未实现     | ROI Pooling |
| ROIAlign             | ❌        | ❌        | ❌ 未实现     | ROI Align |
| Scale                | ❌        | ❌        | ❌ 未实现     | 缩放 |
| SDPA                 | ❌        | ❌        | ❌ 未实现     | Scaled Dot-Product Attention |
| SELU                 | ❌        | ❌        | ❌ 未实现     | SELU 激活 |
| Shrink               | ❌        | ❌        | ❌ 未实现     | Shrink |
| ShuffleChannel       | ❌        | ❌        | ❌ 未实现     | 通道打散 |
| Sigmoid              | ❌        | ❌        | ❌ 未实现     | Sigmoid 激活 |
| Slice                | ❌        | ❌        | ❌ 未实现     | Slice |
| Softmax              | ❌        | ❌        | ❌ 未实现     | Softmax |
| Softplus             | ❌        | ❌        | ❌ 未实现     | Softplus |
| Spectrogram          | ❌        | ❌        | ❌ 未实现     | 频谱 |
| Squeeze              | ❌        | ❌        | ❌ 未实现     | 删除维度 |
| Swish                | ❌        | ❌        | ❌ 未实现     | Swish |
| TanH                 | ❌        | ❌        | ❌ 未实现     | 双曲正切 |
| Tile                 | ❌        | ❌        | ❌ 未实现     | 重复制 |
| Unfold               | ❌        | ❌        | ❌ 未实现     | 展开 |
| Yolov3DetectionOutput| ❌        | ❌        | ❌ 未实现     | YOLOv3 检测输出 |
