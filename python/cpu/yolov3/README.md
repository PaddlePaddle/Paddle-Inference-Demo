# 运行 YOLOv3 图像检测样例


## 一：准备环境

请您在环境中安装2.0或以上版本的 Paddle，具体的安装方式请参照[飞桨官方页面](https://www.paddlepaddle.org.cn/)的指示方式。


## 二：下载模型以及测试数据


1）**获取预测模型**

点击[链接](https://paddle-inference-dist.bj.bcebos.com/Paddle-Inference-Demo/yolov3_r50vd_dcn_270e_coco.tgz)下载模型，如果你想获取更多的**模型训练信息**，请访问[这里](https://github.com/PaddlePaddle/PaddleDetection)。解压后存储到该工程的根目录。

```
wget https://paddle-inference-dist.bj.bcebos.com/Paddle-Inference-Demo/yolov3_r50vd_dcn_270e_coco.tgz
tar xzf yolov3_r50vd_dcn_270e_coco.tgz
```


2）**获取预测样例图片**

下载[样例图片](https://paddle-inference-dist.bj.bcebos.com/inference_demo/images/kite.jpg)。

图片如下：
<p align="left">
    <br>
<img src='https://paddle-inference-dist.bj.bcebos.com/inference_demo/images/kite.jpg' width = "200" height = "200">
    <br>
<p>


## 三：运行预测

- 文件`utils.py`包含了图像的预处理等帮助函数。
- 文件`infer_yolov3.py` 包含了创建predictor，读取示例图片，预测，获取输出的等功能。

### 使用 oneDNN 运行样例

```
python infer_yolov3.py --model_file=./yolov3_r50vd_dcn_270e_coco/model.pdmodel --params_file=./yolov3_r50vd_dcn_270e_coco/model.pdiparams
```

输出结果如下所示：

```
category id is 0.0, bbox is [216.26059 697.699   268.60815 848.7649 ]
category id is 0.0, bbox is [113.00742 614.51337 164.59525 762.8045 ]
category id is 0.0, bbox is [ 82.81181 507.96368 103.27139 565.0893 ]
category id is 0.0, bbox is [346.4539  485.327   355.62698 502.63412]
category id is 0.0, bbox is [520.77747 502.9539  532.1869  527.12494]
category id is 0.0, bbox is [ 38.75421 510.04153  53.91417 561.62244]
category id is 0.0, bbox is [ 24.630651 528.03186   36.35131  551.4408  ]
category id is 0.0, bbox is [537.8204 516.3991 551.4925 532.4528]
category id is 0.0, bbox is [176.29276 538.46545 192.09549 572.6228 ]
category id is 0.0, bbox is [1207.4629   452.27505 1214.8047   461.21774]
category id is 33.0, bbox is [593.3794    80.178375 668.2346   151.84273 ]
category id is 33.0, bbox is [467.13992 339.5424  484.5012  358.15115]
category id is 33.0, bbox is [278.30582 236.12378 304.95267 280.59497]
category id is 33.0, bbox is [1082.6643   393.12796 1099.5437   421.86935]
category id is 33.0, bbox is [302.35004 376.8052  320.6112  410.01248]
category id is 33.0, bbox is [575.6267 343.2629 601.619  369.2695]
```

<p align="left">
    <br>
<img src='https://paddle-inference-dist.bj.bcebos.com/inference_demo/images/kite_res.jpg' width = "200" height = "200">
    <br>
<p>

## 相关链接
- [Paddle Inference使用Quick Start！]()
- [Paddle Inference Python Api使用]()
