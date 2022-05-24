# Python 推理部署

本文主要介绍 Paddle Inferrence Python API 的安装。主要分为以下三个章节：环境准备，安装步骤，和验证安装。

## 环境准备

- Python: 3.6/3.7/3.8/3.9
- CUDA 10.1 / CUDA 10.2 / CUDA 11.0 / CUDA 11.1 / CUDA 11.2, cudnn 7+ （仅在使用GPU版本的预测库时需要）


## 开始安装

请参照 [官方主页-快速安装](https://www.paddlepaddle.org.cn/install/quick) 页面进行自行安装或编译，当前支持 pip/conda 安装，docker 镜像以及[源码编译]()等多种方式来准备 Paddle Inference 开发环境。

- CPU
```
python -m pip install paddlepaddle==2.3.0 -i https://mirror.baidu.com/pypi/simple
```
- GPU(以CUDA11.0版本为例)
```
python -m pip install paddlepaddle-gpu==2.3.0.post110 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
```

- GPU(TensorRT版本安装)

## 验证安装

终端输入 `paddle`, 出现版本提示信息则表示安装成功

```
PaddlePaddle 2.3.0, compiled with
    with_avx: ON
    with_gpu: OFF
    with_mkl: OFF
    with_mkldnn: OFF
    with_python: ON
```

## 使用方法
