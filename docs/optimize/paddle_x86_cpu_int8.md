# X86 CPU 上部署量化模型

## 1 概述

众所周知，模型量化可以有效加快模型预测性能，飞桨也提供了强大的模型量化功能。所以，本文主要介绍在X86 CPU部署PaddleSlim产出的量化模型。

对于常见图像分类模型，在Casecade Lake机器上（例如Intel® Xeon® Gold 6271、6248，X2XX等），INT8模型进行推理的速度通常是FP32模型的3-3.7倍；在SkyLake机器上（例如Intel® Xeon® Gold 6148、8180，X1XX等），INT8模型进行推理的速度通常是FP32模型的1.5倍。

X86 CPU部署量化模型的步骤：
* 产出量化模型：使用PaddleSlim训练并产出量化模型
* 转换量化模型：将量化模型转换成最终部署的量化模型
* 部署量化模型：使用Paddle Inference预测库部署量化模型

## 2 图像分类INT8模型在 Xeon(R) 6271 上的精度和性能

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


## 自然语言处理INT8模型 Ernie, GRU, LSTM 模型在 Xeon(R) 6271 上的性能和精度

>**自然语言处理INT8模型 Ernie, GRU, LSTM 模型在 Xeon(R) 6271 上的性能**

|     Ernie Latency      | FP32 Latency (ms) | INT8 Latency (ms) | Ratio (FP32/INT8) |
| :--------------: | :---------------: | :---------------: | :---------------: |
|  Ernie 1 thread  |      237.21       |       79.26       |       2.99X       |
| Ernie 20 threads |       22.08       |       12.57       |       1.76X       |

| GRU Performance (QPS)              | Naive FP32 | INT88 | Int8/Native FP32 |
| ------------------------------ | ---------- | ----- | ---------------- |
| GRU bs 1, thread 1             | 1108       | 1393  | 1.26             |
| GRU repeat 1, bs 50, thread 1  | 2175       | 3199  | 1.47             |
| GRU repeat 10, bs 50, thread 1 | 2165       | 3334  | 1.54             |

| LSTM Performance (QPS) |  FP32   |  INT8   | INT8 /FP32 |
| :---------------: | :-----: | :-----: | :--------: |
|   LSTM 1 thread   | 4895.65 | 7190.55 |    1.47    |
|  LSTM 4 threads   | 6370.86 | 7942.51 |    1.25    |


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

## 3 PaddleSlim 产出量化模型

X86 CPU预测端支持PaddleSlim量化训练方法和静态离线量化方法产出的量化模型。

关于使用PaddleSlim产出量化模型，请参考文档：
* [静态离线量化-快速开始](https://paddleslim.readthedocs.io/zh_CN/latest/quick_start/quant_post_static_tutorial.html)
* [量化训练-快速开始](https://paddleslim.readthedocs.io/zh_CN/latest/quick_start/quant_aware_tutorial.html)
* [目标检测模型量化](https://paddleslim.readthedocs.io/zh_CN/latest/tutorials/paddledetection_slim_quantization_tutorial.html)
* [量化API文档](https://paddleslim.readthedocs.io/zh_CN/latest/api_cn/quantization_api.html)

在产出部署在X86 CPU预测端的模型时，需要注意：
* 静态离线量化方法支持的量化OP有conv2d, depthwise_conv2d, mul和matmul，所以 `quant_post_static`的输入参数 `quantizable_op_type`可以是这四个op的组合。
* 量化训练方法支持的量化OP有conv2d, depthwise_conv2d, mul和matmul，所以 `quant_aware` 输入配置config中的`quantize_op_types`可以是这四个op的组合。


## 4 转换量化模型

在X86 CPU预测端上部署量化模型之前，需要对量化模型进行转换和优化操作。

### 安装Paddle

参考[Paddle官网](https://www.paddlepaddle.org.cn/)，安装Paddle最新CPU或者GPU版本。

### 准备脚本

下载[脚本](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/contrib/slim/tests/save_quant_model.py)到本地.

```
wget https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/contrib/slim/tests/save_quant_model.py
```

save_quant_model.py脚本的参数说明：
* quant_model_path: 为输入参数，必填。为PaddleSlim产出的量化模型。
* int8_model_save_path: 量化模型转换后保存的路径。

### 转换量化模型

使用脚本转化量化模型，比如：

```
python save_quant_model.py \
    --quant_model_path=/PATH/TO/SAVE/FLOAT32/QUANT/MODEL \
    --int8_model_save_path=/PATH/TO/SAVE/INT8/MODEL
```

## 5 Paddle Inference 部署量化模型

### 检查机器

* 大家可以通过在命令行输入`lscpu`查看本机支持指令。
* 在支持avx512_vnni的CPU服务器上，如：Casecade Lake, Model name: Intel(R) Xeon(R) Gold X2XX，INT8精度和性能最高，INT8性能提升为FP32模型的3~3.7倍。
* 在支持avx512但是不支持avx512_vnni的CPU服务器上，如：SkyLake, Model name：Intel(R) Xeon(R) Gold X1XX，INT8性能为FP32性能的1.5倍左右。
* 请确保机器支持完整的avx512指令集。

### 预测部署

参考[X86 Linux上预测部署示例](../demo_tutorial/x86_linux_demo)和[X86 Windows上预测部署示例](../demo_tutorial/x86_windows_demo)，准备预测库，对模型进行部署。

请注意，在X86 CPU预测端部署量化模型，必须开启MKLDNN，不要开启IrOptim。

C++ API举例如下。

```c++
paddle_infer::Config config;
if (FLAGS_model_dir == "") {
config.SetModel(FLAGS_model_file, FLAGS_params_file); // Load combined model
} else {
config.SetModel(FLAGS_model_dir); // Load no-combined model
}
config.EnableMKLDNN();
config.SwitchIrOptim(false);
config.SetCpuMathLibraryNumThreads(FLAGS_threads);
config.EnableMemoryOptim();

auto predictor = paddle_infer::CreatePredictor(config);
```

Python API举例如下。

```python
if args.model_dir == "":
    config = Config(args.model_file, args.params_file)
else:
    config = Config(args.model_dir)
config.enable_mkldnn()
config.set_cpu_math_library_num_threads(args.threads)
config.switch_ir_optim(False)
config.enable_memory_optim()

predictor = create_predictor(config)
```