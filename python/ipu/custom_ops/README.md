# 自定义算子 Python 环境使用示例

## 一：简介
本文档说明了如何在 Python 环境执行飞桨，加载自定义算子文件，执行推理。

## 二：获取本样例中的自定义算子模型

下载地址：https://paddle-inference-dist.bj.bcebos.com/inference_demo/custom_operator/custom_relu_infer_model.tgz

执行 `tar -zxvf custom_relu_infer_model.tgz` 将模型文件解压至当前目录。

## 三：样例运行

文件 `custom_relu_op.cc`、`custom_relu_op_ipu.cc` 为自定义算子源文件，自定义算子编写方式请参考[飞桨官网文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/07_new_op/new_custom_op.html)。

文件`infer.py` 为训练和预测的样例程序。  

执行命令：

```
python infer.py
```

## 三：注意事项

1、请留意 Python 环境安装的飞桨版本。自定义算子功能的正确训练和推理建议使用 飞桨 2.3 及以上版本；使用 IPU 功能需要安装 IPU 版本的飞桨。

## 更多链接
- [Paddle Inference使用Quick Start！](https://paddle-inference.readthedocs.io/en/latest/introduction/quick_start.html)
- [Paddle Inference C++ Api使用](https://paddle-inference.readthedocs.io/en/latest/api_reference/cxx_api_index.html)
- [Paddle Inference Python Api使用](https://paddle-inference.readthedocs.io/en/latest/api_reference/python_api_index.html)
