## 运行YOLOv3图像检测样例


### 一：准备环境

请您在环境中安装1.7或以上版本的Paddle，具体的安装方式请参照[飞桨官方页面](https://www.paddlepaddle.org.cn/)的指示方式。


### 二：下载模型以及测试数据


1）**获取预测模型**

点击[链接](https://paddle-inference-dist.cdn.bcebos.com/PaddleLite/yolov3_infer.tar.gz)下载模型， 该模型在imagenet数据集训练得到的，如果你想获取更多的**模型训练信息**，请访问[这里](https://github.com/PaddlePaddle/PaddleDetection)。


2）**获取预测样例图片**

下载[样例图片](https://paddle-inference-dist.bj.bcebos.com/inference_demo/python/resnet50/ILSVRC2012_val_00000247.jpeg)。

图片如下：
<p align="left">
    <br>
<img src='https://paddle-inference-dist.bj.bcebos.com/inference_demo/python/resnet50/ILSVRC2012_val_00000247.jpeg' width = "200" height = "200">
    <br>
<p>


### 三：运行预测

文件`utils.py`包含了图像的预处理等帮助函数。
文件`infer_yolov3.py` 包含了创建predictor，读取示例图片，预测，获取输出的等功能。

运行：
```
python infer_yolov3.py --model_file=./yolov3_infer/__model__ --params_file=./yolov3_infer/__params__ --use_gpu=1
```

运行的结果为： ('category id is ', 14.0, ', bbox is ', array([120.713684, 118.58473 , 420.50403 , 558.6027  ], dtype=float32))。
14表示图片的类别。我们通过imagenet [类别映射表](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a)， 可以找到对应的类别，bbox则为检测到的物体框，该示例会将物体框画到图像上并存储res.jpg文件到本地目录。

### 相关链接
- [Paddle Inference使用Quick Start！]()
- [Paddle Inference Python Api使用]()

