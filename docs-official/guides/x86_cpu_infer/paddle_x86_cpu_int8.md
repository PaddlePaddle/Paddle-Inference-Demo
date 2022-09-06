# X86 CPU 上部署量化模型

## 概述

本文主要介绍在 X86 CPU 部署 PaddleSlim 产出的量化模型。

X86 CPU部署量化模型的步骤：
- [产出量化模型](#产出量化模型)：使用 PaddleSlim 训练并产出量化模型
- [部署量化模型](#部署量化模型)：使用 Paddle Inference 预测库部署量化模型
- [性能 benchmark](#性能benchmark): 部署量化模型的性能数据


## 产出量化模型

X86 CPU 预测端支持 PaddleSlim 量化训练方法和静态离线量化方法产出的量化模型。

关于使用 PaddleSlim 产出量化模型，请参考文档：
- 静态图量化
  - [离线量化-快速开始](https://paddleslim.readthedocs.io/zh_CN/latest/quick_start/static/quant_post_static_tutorial.html)
  - [量化训练-快速开始](https://paddleslim.readthedocs.io/zh_CN/latest/quick_start/static/quant_aware_tutorial.html)
  - [量化 API 文档](https://paddleslim.readthedocs.io/zh_CN/latest/api_cn/static/quant/quantization_api.html)
- 动态图量化
  - [离线量化-快速开始](https://paddleslim.readthedocs.io/zh_CN/latest/quick_start/dygraph/dygraph_quant_post_tutorial.html)
  - [量化训练-快速开始](https://paddleslim.readthedocs.io/zh_CN/latest/quick_start/dygraph/dygraph_quant_aware_training_tutorial.html)
  - [量化 API 文档](https://paddleslim.readthedocs.io/zh_CN/latest/api_cn/dygraph/quanter/qat.html)


在产出部署在 X86 CPU 预测端的模型时，需要注意：
* 静态离线量化方法支持的量化 OP 有 conv2d, depthwise_conv2d, mul 和 matmul，所以 `quant_post_static` 的输入参数  `quantizable_op_type` 可以是这四个op的组合。
* 量化训练方法支持的量化 OP 有 conv2d, depthwise_conv2d, mul 和 matmul，所以 `quant_aware` 输入配置config中的`quantize_op_types` 可以是这四个 op 的组合。

## 部署量化模型

### 检查机器

* 大家可以通过在命令行输入 `lscpu` 查看本机支持指令。
* 在支持 avx512_vnni 的 CPU 服务器上，如：Casecade Lake, Model name: Intel(R) Xeon(R) Gold X2XX，INT8 精度和性能最高，INT8 性能提升为 FP32 模型的 3~3.7 倍。
* 在支持 avx512 但是不支持 avx512_vnni 的 CPU 服务器上，如：SkyLake, Model name：Intel(R) Xeon(R) Gold X1XX，INT8 性能为 FP32 性能的 1.5 倍左右。
* 请确保机器支持完整的 avx512 指令集。

### 预测部署

参考[C++ 预测部署示例](../../../c++/cpu/resnet50)和[Python 预测部署示例](../../../pythoncpu/resnet50)，准备预测库，对模型进行部署。

**请注意：**
- 在 X86 CPU 预测端部署量化模型，必须开启 MKLDNN, MKLDNNINT8 和 IrOptim。
- 新版本量化模型还需要使用 SetCalibrationFilePath 设置量化模型的 calibration 文件路径
- 生成量化模型后，可以使用如下命令部署量化模型

用户可用以下提前准备的量化模型进行验证
```bash
wget https://paddle-slim-models.bj.bcebos.com/act/ResNet50_vd_QAT.tar
tar -xf ResNet50_vd_QAT.tar
```

C++ 部署示例中运行量化模型的命令如下：

```bash
./build/resnet50_test --model_file ResNet50_vd_QAT/inference.pdmodel --params_file ResNet50_vd_QAT/inference.pdiparams --calibration_file ResNet50_vd_QAT/calibration_table.txt
```

Python 部署示例中运行量化模型的命令如下：

```bash
python infer_resnet.py --model_file=ResNet50_vd_QAT/inference.pdmodel --params_file ResNet50_vd_QAT/inference.pdiparams --calibration_file ResNet50_vd_QAT/calibration_table.txt
```

## 性能benchmark

对于常见图像分类模型，在Casecade Lake机器上（例如Intel® Xeon® Gold 6271、6248，X2XX等），图片分类模型INT8模型预测性能可达FP32模型的3-3.7倍, 自然语言处理模型INT8模型预测性能可达到FP32的1.5-3倍；在SkyLake机器上（例如Intel® Xeon® Gold 6148、8180，X1XX等），图片分类INT8模型预测性能可达FP32模型的1.5倍左右。

### 图像分类INT8模型在 Xeon(R) 6271 上的精度和性能

>**图像分类INT8模型在 Intel(R) Xeon(R) Gold 6271 上精度**

|     Model    | FP32 Top1 Accuracy | INT8 Top1 Accuracy | Top1 Diff | FP32 Top5 Accuracy | INT8 Top5 Accuracy | Top5 Diff |
|:------------:|:------------------:|:------------------:|:---------:|:------------------:|:------------------:|:---------:|
| MobileNet-V1 |       70.78%       |       70.74%       |   -0.04%  |       89.69%       |       89.43%       |   -0.26%  |
| MobileNet-V2 |       71.90%       |       72.21%       |   0.31%   |       90.56%       |       90.62%       |   0.06%   |
|   ResNet101  |       77.50%       |       77.60%       |   0.10%   |       93.58%       |       93.55%       |   -0.03%  |
|   ResNet50   |       76.63%       |       76.50%       |   -0.13%  |       93.10%       |       92.98%       |   -0.12%  |
|     VGG16    |       72.08%       |       71.74%       |   -0.34%  |       90.63%       |       89.71%       |   -0.92%  |
|     VGG19    |       72.57%       |       72.12%       |   -0.45%  |       90.84%       |       90.15%       |   -0.69%  |

>**图像分类INT8模型在 Intel(R) Xeon(R) Gold 6271 单核上性能**

|     Model    | FP32 (images/s) | INT8 (images/s) | Ratio (INT8/FP32) |
|:------------:|:---------------:|:---------------:|:-----------------:|
| MobileNet-V1 |      74.05      |      216.36     |        2.92       |
| MobileNet-V2 |      88.60      |      205.84     |        2.32       |
|   ResNet101  |       7.20      |      26.48      |        3.68       |
|   ResNet50   |      13.23      |      50.02      |        3.78       |
|     VGG16    |       3.47      |      10.67      |        3.07       |
|     VGG19    |       2.83      |       9.09      |        3.21       |

- `Accuracy`: 准确率
- `images/s`: 每秒推理的图片数量

### 自然语言处理INT8模型 Ernie, GRU, LSTM 模型在 Xeon(R) 6271 上的性能和精度

>**自然语言处理INT8模型 Ernie, GRU, LSTM 模型在 Xeon(R) 6271 上的性能**

|     Ernie Latency      | FP32 Latency (ms) | INT8 Latency (ms) | Ratio (FP32/INT8) |
| :--------------: | :---------------: | :---------------: | :---------------: |
|  Ernie 1 thread  |      237.21       |       79.26       |       2.99X       |
| Ernie 20 threads |       22.08       |       12.57       |       1.76X       |

| GRU Performance (QPS)              | Naive FP32 | INT8 | Int8/Native FP32 |
| ------------------------------ | ---------- | ----- | ---------------- |
| GRU bs 1, thread 1             | 1108       | 1393  | 1.26             |
| GRU repeat 1, bs 50, thread 1  | 2175       | 3199  | 1.47             |
| GRU repeat 10, bs 50, thread 1 | 2165       | 3334  | 1.54             |

| LSTM Performance (QPS) |  FP32   |  INT8   | INT8 /FP32 |
| :---------------: | :-----: | :-----: | :--------: |
|   LSTM 1 thread   | 4895.65 | 7190.55 |    1.47    |
|  LSTM 4 threads   | 6370.86 | 7942.51 |    1.25    |

- `ms`: 毫秒
- `QPS`: 每秒执行的推理次数

>**自然语言处理INT8模型 Ernie, GRU, LSTM 模型在 Xeon(R) 6271 上的精度**

|  Ernie   | FP32 Accuracy | INT8 Accuracy | Accuracy Diff |
| :------: | :-----------: | :-----------: | :-----------: |
| accuracy |    80.20%     |    79.44%     |    -0.76%     |

| LAC (GRU) | FP32    | INT8    | Accuracy diff |
| --------- | ------- | ------- | ------------- |
| accuracy  | 0.89326 | 0.89323 | -0.00007      |

|  LSTM   | FP32  | INT8  |
| :-----: | :---: | :---: |
| HX_ACC  | 0.933 | 0.925 |
| CTC_ACC | 0.999 | 1.000 |


**Note:**
* 图像分类复现 demo 可参考 [Intel CPU量化部署图像分类模型](https://github.com/PaddlePaddle/PaddleSlim/tree/develop/demo/mkldnn_quant)
* Ernie 复现 demo 可参考 [ERNIE QAT INT8 精度与性能复现](https://github.com/PaddlePaddle/benchmark/tree/master/Inference/c%2B%2B/ernie/mkldnn)
* LAC (GRU) 复现 demo 可参考 [GRU INT8 精度与性能复现](https://github.com/PaddlePaddle/Paddle-Inference-Demo/tree/master/c%2B%2B/x86_gru_int8)
* LSTM 复现 demo 可参考 [LSTM INT8 精度与性能复现](https://github.com/PaddlePaddle/Paddle-Inference-Demo/tree/master/python/x86_lstm_demo)




