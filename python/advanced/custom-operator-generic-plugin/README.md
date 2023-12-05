# 自定义算子通用Plugin使用示例（Python）

## 一：简介
本文档说明了如何在 Python 环境中实现自定义算子自动生成TensorRT Plugin。

## 二：样例运行

文件 `custom_gap_op.cc`、`custom_gap_op.cu` 为自定义算子源文件，自定义算子编写方式请参考[飞桨官网文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/07_new_op/new_custom_op.html)。

执行命令：

```
pip install paddlepaddle-gpu==2.6.0
bash run.sh
```

## 三：注意事项

1. 请留意 Python 环境安装的飞桨版本。自定义算子自动生成TensorRT plugin的功能需要 飞桨 2.6 及以上版本；使用 GPU 功能需要安装 GPU 版本的飞桨。

2. 自定义算子相关文件的写法参考 [自定义算子通用Plugin使用示例(C++)](../../../c%2B%2B/advanced/custom-operator-generic-plugin/README.md)


## 更多链接
- [Paddle Inference使用Quick Start！](https://paddle-inference.readthedocs.io/en/latest/introduction/quick_start.html)
- [Paddle Inference C++ Api使用](https://paddle-inference.readthedocs.io/en/latest/api_reference/cxx_api_index.html)
- [Paddle Inference Python Api使用](https://paddle-inference.readthedocs.io/en/latest/api_reference/python_api_index.html)
