# 运行 Resnet50 图像分类样例

ResNet50 样例展示了单输入模型在 GPU 下的推理过程。运行步骤如下：

## 一：准备环境

请您在环境中安装2.0或以上版本的Paddle，具体的安装方式请参照[飞桨官方页面](https://www.paddlepaddle.org.cn/)的指示方式。


## 二：下载模型以及测试数据


1）**获取预测模型**

下载[模型](https://paddle-inference-dist.bj.bcebos.com/Paddle-Inference-Demo/resnet50.tgz)，模型为imagenet 数据集训练得到的，如果你想获取更多的模型训练信息，请访问[这里](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/image_classification)。解压后存储到该工程的根目录。


2）**获取预测样例图片**

下载[样例图片](https://paddle-inference-dist.bj.bcebos.com/inference_demo/python/resnet50/ILSVRC2012_val_00000247.jpeg)。

图片如下：
<p align="left">
    <br>
<img src='https://paddle-inference-dist.bj.bcebos.com/inference_demo/python/resnet50/ILSVRC2012_val_00000247.jpeg' width = "200" height = "200">
    <br>
<p>


## 三：运行预测

- 文件`img_preprocess.py`包含了图像的预处理。    
- 文件`infer_resnet.py` 包含了创建predictor，读取示例图片，预测，获取输出的等功能。

### 使用原生 GPU 运行样例

```shell
python infer_resnet.py --model_file=./resnet50/inference.pdmodel --params_file=./resnet50/inference.pdiparams 
```
### 使用 GPU混合精度推理 运行样例

```shell
python infer_resnet.py --model_file=./resnet50/inference.pdmodel --params_file=./resnet50/inference.pdiparams --run_mode=gpu_fp16 
```

### 使用 Trt Fp32 运行样例

```shell
python infer_resnet.py --model_file=./resnet50/inference.pdmodel --params_file=./resnet50/inference.pdiparams --run_mode=trt_fp32
```

### 使用 Trt Fp16 运行样例

```shell
python infer_resnet.py --model_file=./resnet50/inference.pdmodel --params_file=./resnet50/inference.pdiparams --run_mode=trt_fp16
```

### 使用 Trt Int8 运行样例

在使用 Trt In8 运行样例时，相同的运行命令需要执行两次。

#### 生成量化校准表

```shell
python infer_resnet.py --model_file=./resnet50/inference.pdmodel --params_file=./resnet50/inference.pdiparams --run_mode=trt_int8
```

生成校准表的log：
```
I0623 08:40:49.386909 107053 tensorrt_engine_op.h:159] This process is generating calibration table for Paddle TRT int8...
I0623 08:40:49.387279 107057 tensorrt_engine_op.h:352] Prepare TRT engine (Optimize model structure, Select OP kernel etc). This process may cost a lot of time.
I0623 08:41:13.784473 107053 analysis_predictor.cc:791] Wait for calib threads done.
I0623 08:41:14.419198 107053 analysis_predictor.cc:793] Generating TRT Calibration table data, this may cost a lot of time...
```

执行后，模型文件夹`ResNet50`下的`_opt_cache`文件夹下会多出一个名字为`trt_calib_*`的文件，即校准表。

#### 加载校准表执行预测

```shell
python infer_resnet.py --model_file=./resnet50/inference.pdmodel --params_file=./resnet50/inference.pdiparams --run_mode=trt_int8
```

加载校准表预测的log：
```
I0623 08:40:27.217701 107040 tensorrt_subgraph_pass.cc:258] RUN Paddle TRT int8 calibration mode...
I0623 08:40:27.217834 107040 tensorrt_subgraph_pass.cc:321] Prepare TRT engine (Optimize model structure, Select OP kernel etc). This process may cost a lot of time.
```

### 使用 Trt dynamic shape 运行样例（以 Fp32 为例）

```shell
python infer_resnet.py --model_file=./resnet50/inference.pdmodel --params_file=./resnet50/inference.pdiparams --run_mode=trt_fp32 --use_dynamic_shape=1
```

运行的结果为： ('class index: ', 13)。
13表示图片的类别。我们通过imagenet [类别映射表](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a)， 可以找到对应的类别，即junco, snowbird，由此说明我们的分类器分类正确。

**提示**: 如果想使用onnxruntime后端，可以在python运行命令中传入`--use_onnxruntime=1`参数

### 相关链接
- [Paddle Inference Python Api使用](https://paddle-inference.readthedocs.io/en/latest/api_reference/python_api_index.html)

