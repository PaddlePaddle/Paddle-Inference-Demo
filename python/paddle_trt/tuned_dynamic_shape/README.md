## 使用 Paddle-TRT TunedDynamicShape 能力

该文档为使用 Paddle-TRT TunedDynamicShape 的实践demo。如果您刚接触Paddle-TRT，推荐先访问[这里](https://paddle-inference.readthedocs.io/en/latest/optimize/paddle_trt.html)对Paddle-TRT有个初步认识。

### 准备环境

请您在环境中安装2.0或以上版本的Paddle，具体的安装方式请参照[飞桨官方页面](https://www.paddlepaddle.org.cn/)的指示方式。

### 下载测试模型

下载[模型](https://paddle-inference-dist.bj.bcebos.com/Paddle-Inference-Demo/resnet50.tgz)，模型为imagenet 数据集训练得到的，如果你想获取更多的模型训练信息，请访问[这里](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/image_classification)。解压后存储到该工程的根目录。

#### TunedDynamicShape测试

**1、首先您需要针对业务数据进行离线tune，来获取计算图中所有中间tensor的shape范围，并将其存储在config中配置的shape_range_info.pbtxt文件中**

```
python infer_tune.py --model_file ./resnet50/inference.pdmodel --params_file ./resnet50/inference.pdiparams --tune 1
```

**2、有了离线tune得到的shape范围信息后，您可以使用该文件自动对所有的trt子图设置其输入的shape范围。**

```
python infer_tune.py --model_file ./resnet50/inference.pdmodel --params_file ./resnet50/inference.pdiparams --use_gpu 1 --use_trt 1 --tuned_dynamic_shape 1
```

### 更多链接
- [Paddle Inference使用Quick Start！](https://paddle-inference.readthedocs.io/en/latest/introduction/quick_start.html)
- [Paddle Inference C++ Api使用](https://paddle-inference.readthedocs.io/en/latest/user_guides/cxx_api.html)
- [Paddle Inference Python Api使用](https://paddle-inference.readthedocs.io/en/latest/user_guides/inference_python_api.html)
