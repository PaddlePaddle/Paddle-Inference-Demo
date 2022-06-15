# GPU TensorRT 低精度或量化推理

深度学习模型训练好之后，其权重参数在一定程度上是冗余的，在很多任务上，我们可以采用低精度或量化进行模型推理而不影响模型精度。这一方面可以减少访存、提升计算效率，另一方面，可以降低显存占用。Paddle Inference 的 GPU 原生推理仅支持 Fp32，Fp16 精度目前处于实验阶段；采用 TensorRT 加速推理的方式可支持 Fp32、Fp16 以及 Int8 量化推理。使用前，请参考[链接](https://docs.nvidia.com/deeplearning/tensorrt/support-matrix/index.html#hardware-precision-matrix)确保您的 GPU 硬件支持您使用的精度。


- [1. Fp16 推理](#1)
- [2. Int8 量化推理](#2)
- [Int8 量化推理的完整 demo](https://github.com/PaddlePaddle/Paddle-Inference-Demo/tree/master/c%2B%2B/gpu/resnet50)

<a name="1"></a>

## 1. Fp16 推理

为使用 Fp16 带来的性能提升，只需在指定 TensorRT 配置时，将 **precision_mode** 设为 **paddle_infer.PrecisionType.Half**即可，示例如下：

```python
	config.enable_tensorrt_engine(
		workspace_size = 1<<30,
		max_batch_size=1, min_subgraph_size=5,
		precision_mode=paddle_infer.PrecisionType.Half,
		use_static=False, use_calib_mode=False)
```

<a name="2"></a>

## 2. Int8 量化推理

使用 Int8 量化推理的流程可以分为两步：（1）产出量化模型。（2）加载量化模型进行推理。下面我们对使用Paddle Inference 进行 Int8 量化推理的完整流程进行详细介绍。

**1. 产出量化模型**

目前，我们支持通过两种方式产出量化模型：

a. 使用 TensorRT 自带的 Int8 离线量化校准功能。首先基于训练好的 Fp32 模型和少量校准数据（如 500～1000 张图片）生成校准表（Calibration table）。然后推理时，加载 Fp32 模型和此校准表即可使用 Int8 精度推理。生成校准表的方法如下：

  - 指定 TensorRT 配置时，将 **precision_mode** 设为 **paddle_infer.PrecisionType.Int8** 并且设置 **use_calib_mode** 为 **True**。

```python
      config.enable_tensorrt_engine(
        workspace_size=1<<30,
        max_batch_size=1, min_subgraph_size=5,
        precision_mode=paddle_infer.PrecisionType.Int8,
        use_static=False, use_calib_mode=True)
```
准备 500 张左右的真实输入数据，在上述配置下，运行模型。（ TensorRT 会统计模型中每个 tensor 值的范围信息，并将其记录到校准表中，运行结束后，会将校准表写入模型目录下的 `_opt_cache` 目录中）。

如果想要了解使用 TensorRT 自带 Int8 离线量化校准功能生成校准表的完整代码，请参考[链接](https://github.com/PaddlePaddle/Paddle-Inference-Demo/tree/master/c%2B%2B/gpu/resnet50)。

b. 使用模型压缩工具库 PaddleSlim 产出量化模型。PaddleSlim 支持离线量化和在线量化功能，其中，离线量化与TensorRT 离线量化校准原理相似；在线量化又称量化训练(Quantization Aware Training, QAT)，是基于较多数据（如>=5000张图片）对预训练模型进行重新训练，使用模拟量化的思想，在训练阶段更新权重，实现减小量化误差的方法。使用PaddleSlim产出量化模型可以参考文档：
  
  - 离线量化 [快速开始教程](https://github.com/PaddlePaddle/PaddleSlim/blob/release/2.3/docs/zh_cn/quick_start/static/quant_post_static_tutorial.md)
  - 离线量化 [API 接口说明](https://github.com/PaddlePaddle/PaddleSlim/blob/release/2.3/docs/zh_cn/api_cn/static/quant/quantization_api.rst)
  - 离线量化 [Demo](https://github.com/PaddlePaddle/PaddleSlim/tree/release/2.3/demo/quant/quant_post)
  - 量化训练 [快速开始教程](https://github.com/PaddlePaddle/PaddleSlim/blob/release/2.3/docs/zh_cn/quick_start/dygraph/dygraph_quant_aware_training_tutorial.md)
  - 量化训练 [API 接口说明](https://github.com/PaddlePaddle/PaddleSlim/blob/release/2.3/docs/zh_cn/api_cn/dygraph/quanter/qat.rst)
  - 量化训练 [Demo](https://github.com/PaddlePaddle/PaddleSlim/tree/release/2.3/demo/quant/quant_aware)

离线量化的优点是无需重新训练，简单易用，但量化后精度可能受影响；量化训练的优点是模型精度受量化影响较小，但需要重新训练模型，使用门槛稍高。在实际使用中，我们推荐先使用 TensorRT 离线量化校准功能生成量化模型，若精度不能满足需求，再使用 PaddleSlim 产出量化模型。

**2. 加载量化模型进行Int8推理**       


加载量化模型进行 Int8 推理，需要在指定 TensorRT 配置时，将 **precision_mode** 设置为 **paddle_infer.PrecisionType.Int8** 。

若使用的量化模型为 TensorRT 离线量化校准产出的，需要将 **use_calib_mode** 设为 **True** ：

```python
    config.enable_tensorrt_engine(
      workspace_size=1<<30,
      max_batch_size=1, min_subgraph_size=5,
      precision_mode=paddle_infer.PrecisionType.Int8,
      use_static=False, use_calib_mode=True)
```

若使用的量化模型为 PaddleSlim 量化产出的，需要将 **use_calib_mode** 设为 **False** ：

```python
    config.enable_tensorrt_engine(
      workspace_size=1<<30,
      max_batch_size=1, min_subgraph_size=5,
      precision_mode=paddle_infer.PrecisionType.Int8,
      use_static=False, use_calib_mode=False)
```

Int8 量化推理的完整 demo 请参考[链接](https://github.com/PaddlePaddle/Paddle-Inference-Demo/tree/master/c%2B%2B/gpu/resnet50)。
