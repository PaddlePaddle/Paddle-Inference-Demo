# 自定义算子 Python 环境使用示例

## 一：简介
本文档说明了如何在 Python 环境执行飞桨，加载自定义算子文件，分别执行模型训练和推理。

## 二：样例运行

文件 `custom_relu_op.cc`、`custom_relu_op.cu` 为自定义算子源文件，自定义算子编写方式请参考[飞桨官网文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/07_new_op/new_custom_op.html)。

文件`train_and_infer.py` 为训练和预测的样例程序。  

执行命令：

```
pip install paddlepaddle-gpu==2.3.0
python train_and_infer.py
```

## 三：注意事项

1、请留意 Python 环境安装的飞桨版本。自定义算子功能的正确训练和推理建议使用 飞桨 2.1 及以上版本；使用 GPU 功能需要安装 GPU 版本的飞桨。

2、本样例训练得到的推理部署模型可以直接使用 C++ 预测器加载运行。在使用 C++ 端的自定义算子时，请参照样例修改自定义算子源文件所含的头文件。

## 更多链接
- [Paddle Inference使用Quick Start！](https://www.paddlepaddle.org.cn/inference/master/guides/quick_start/index_quick_start.html)
- [Paddle Inference C++ Api使用](https://www.paddlepaddle.org.cn/inference/master/api_reference/cxx_api_doc/cxx_api_index.html)
- [Paddle Inference Python Api使用](https://www.paddlepaddle.org.cn/inference/master/api_reference/python_api_doc/python_api_index.html)
