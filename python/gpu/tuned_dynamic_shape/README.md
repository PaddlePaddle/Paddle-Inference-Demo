# 使用 Paddle-TRT TunedDynamicShape 能力

该文档为使用 Paddle-TRT TunedDynamicShape 的实践 demo。如果您刚接触 Paddle-TRT，推荐先访问[这里](https://paddle-inference.readthedocs.io/en/latest/optimize/paddle_trt.html)对 Paddle-TRT 有个初步认识。

## 一：准备环境

请您在环境中安装2.0或以上版本的 Paddle，具体的安装方式请参照[飞桨官方页面](https://www.paddlepaddle.org.cn/)的指示方式。

### 二：下载测试模型

下载[模型](https://paddle-inference-dist.bj.bcebos.com/Paddle-Inference-Demo/resnet50.tgz)，模型为 imagenet 数据集训练得到的，如果你想获取更多的模型训练信息，请访问[这里](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/image_classification)。解压后存储到该工程的根目录。

### 三：运行 TunedDynamicShape 样例

**1、首先您需要针对业务数据进行离线 tune，来获取计算图中所有中间 tensor 的 shape 范围，并将其存储在 config 中配置的 shape_range_info.pbtxt 文件中**

```
python infer_tune.py --model_file ./resnet50/inference.pdmodel --params_file ./resnet50/inference.pdiparams --tune 1
```

**2、有了离线 tune 得到的 shape 范围信息后，您可以使用该文件自动对所有的 trt 子图设置其输入的 shape 范围。**

```
python infer_tune.py --model_file ./resnet50/inference.pdmodel --params_file ./resnet50/inference.pdiparams --use_gpu 1 --use_trt 1 --tuned_dynamic_shape 1
```

## 更多链接
- [Paddle Inference使用Quick Start！](https://paddle-inference.readthedocs.io/en/latest/introduction/quick_start.html)
- [Paddle Inference C++ Api使用](https://paddle-inference.readthedocs.io/en/latest/user_guides/cxx_api.html)
- [Paddle Inference Python Api使用](https://paddle-inference.readthedocs.io/en/latest/user_guides/inference_python_api.html)
