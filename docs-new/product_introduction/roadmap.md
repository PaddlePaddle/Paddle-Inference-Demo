
# Roadmap

## Release Note

详细 Release Note 请参考 [PaddlePadde ReleaseNote](https://github.com/PaddlePaddle/Paddle/releases)，近期重要 Roadmap 如下：

### 2.3.0 / 2.3.0-rc0

- 新增 Java API 和 ONNX Runtime CPU 后端。
- 针对 ERNIE 类结构模型性能深度优化。

### 2.2.2

- 支持 relu、relu6、tanh、sigmoid、pool2d、concat、batch_norm、split、gelu、scale、swish、prelu、clip、reduce_sum、reduce_mean 算子在静态 shape 且2维输入情况下调用 TensorRT 推理。
- 支持mish激活函数调用 TensorRT 推理。

### 2.2.0 / 2.2.0-rc0

- 新增 TensorRT 子图模式下动态 shape 自动配置功能。使用离线 tune 动态 shape 的方式，提升模型被切分成多个 TensorRT 子图场景下动态 shape 的易用性。
- 新增 pool3d 算子支持 TensorRT 推理。
- Go API 重构。
- 升级 oneDNN 版本为 2.3.2，优化 oneDNN 的 cache 机制。
- 增加 TensorRT 8.0 的支持，在将来的某个版本我们会放弃对 TensorRT 6.x 的支持。
- 支持 TensorRT 8.0 稀疏推理，ERNIE 模型变长输入在不同的 batch_size 下性能提升10% - 30%，ResNeXt101_32x4d 模型在不同的 batch_size 下性能提升 10%。

### 2.1.0

- 发布 C API (experimental)， 功能与 C++ API 基本对齐。
- 预测框架python接口接入训练自定义算子。用户在训练过程中加载自定义算子后，即可像框架原生算子那样，通过 PaddlePredictor 直接执行包含此自定义算子的预测模型部署。
- 支持从内存加载模型时TensorRT序列化和反序列化功能。

### 2.0.2

- Paddle-TRT 适配由 Paddle 2.0 训练保存的 ERNIE / BERT 模型。
- 升级 Paddle 的 oneDNN 版本到 oneDNN 2.2，多个模型预测性能有提升。

### 2.0.1
 
- 增加了对采用 per-layer 方式量化的模型 TensorRT 量化预测的支持。
- 新增 API paddle_infer::Config::EnableTensorRtDLA()，支持在开启 TensorRT 的基础上使用 NVIDIA 的硬件加速器 DLA。
- C++ 和 Python 推理接口新增对昆仑 XPU 的原生支持，用户可因此获得更完备的算子种类支持。

### 2.0.0

- 全面升级推理C++ API
 - C++ 接口新增 paddle_infer 命名空间，包含推理相关接口。
 - ZeroCopyTensor 更名为 Tensor，作为推理接口默认输入输出表示方式。
 - 简化 CreatePaddlePredictor 为 CreatePredictor，只保留对 AnalysisConfig 的支持，不再支持其他多种 Config。
 - 新增服务相关的工具类，比如 PredictorPool，便于创建多个 predictor 时使用。

- 模型相关API
 - load_inference_model 和 save_inference_model 两个API迁移到 paddle.static 下，兼容旧接口，提升易用性。
 - 新增 serialize_program, deserialize_program, serialize_persistables, deserialize_persistables, save_to_file, load_from_file 六个API，用来满足用户执行序列化/反序列化 program，序列化/反序列化 params，以及将模型/参数保存到文件，或从文件中加载模型/参数的需求。

- NV GPU 推理相关
 - 新增对 TensorRT 7.1 版本的适配支持。
 - 新增对 Jetson Xavier NX 硬件的适配支持。
 - Paddle-TRT 动态 shape 功能支持 PaddleSlim 量化 Int8 模型。
 - ERNIE 模型在 Nvidia Telsa T4 上使用 Paddle-TRT FP16 推理性能提升 15%。
 - ERNIE 模型在开启 TenorRT 时增加变长输入的支持，带来性能提升 147%。在软件版本 cuda10.1、cudnn 7.6、tensorrt 6.0、OSS 7.2.1，模型 ernie-base-2.0，数据集 QNLI，输入 BatchSize = 32 时，Nvidia Telsa T4 上的性能从 905 sentences/s 提升到 2237 sentences/s。

- X86 CPU 推理相关
 - 添加了对 oneDNN BF16 的支持：支持 conv2d 和 gru bf16 计算，目前支持 resnet50、googlenet、mobilenetv1 和 mobilenetv2 模型的 BF16 预测。
 - 添加了一些oneDNN 算子的版本兼容性支持。
 - oneDNN 升级到 1.6。

- 自定义OP
 - Python端推理新增对用户自定义OP支持。

- 内存 / 显存相关
 - 新增 TryShrinkMemory 接口，通过释放临时 tensor 的方式减少应用内存 / 显存占用。

- 动态图量化模型支持
 - X86 推理支持动态图量化模型。
 - NVIDIA GPU 推理支持动态图量化模型。

## 用户群

.[](../images/user_qq.png)
