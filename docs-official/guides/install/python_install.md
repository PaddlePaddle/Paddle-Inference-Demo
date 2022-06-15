# 安装 Python API

本文主要介绍 Paddle Inference Python API 的安装。主要分为以下三个章节：环境准备、安装步骤和验证安装。三个章节分别说明了安装前的环境要求、安装的具体流程和成功安装后的验证方法。

## 环境准备

- Python: 3.6 / 3.7 / 3.8 / 3.9
- CUDA 10.1 / CUDA 10.2 / CUDA 11.0 / CUDA 11.2, cuDNN7.6+, TensorRT （仅在使用 GPU 版本的推理库时需要）

您可参考 NVIDIA 官方文档了解 CUDA 和 cuDNN 的安装流程和配置方法，请见 [CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)，[cuDNN](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/)，版本对应关系如下表所示：

|CUDA 版本|cuDNN 版本| TensorRT 版本|
|---|---|---|
|10.2|7.6|7|
|11.0|8.0|7|
|11.2|8.2|8|

## 开始安装

如果已经安装 PaddlePaddle，且不需要使用 TensorRT 进行推理加速，可以跳过此步骤，直接使用为训练安装的 PaddlePaddle 版本。

### 方式一：通过 pip 在线安装（不含TensorRT）

- CPU版本
```
python3 -m pip install paddlepaddle==2.3.0 -i https://mirror.baidu.com/pypi/simple
```
- GPU版本（以 CUDA11.0 为例）
```
python3 -m pip install paddlepaddle-gpu==2.3.0.post110 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
```

### 方式二：下载 whl 包（可选 TensorRT）到本地，然后通过 pip 工具安装

- [下载安装 Linux Python installer](download_lib.html#python)
- [下载安装 Windows Python installer](download_lib.html#id4)

### 方式三：源码安装

参考[源码编译](./compile/index_compile.html)文档。

## 验证安装

### 静态验证方式

安装完成后，可以使用 python3 进入python解释器，输入以下指令，出现 `PaddlePaddle is installed successfully! ` ，说明安装成功。

```python
import paddle
paddle.utils.run_check()
```

### 动态验证方式

您可以编写应用代码并测试结果。请参考 [推理示例(Python)](../quick_start/python_demo) 一节。


## 开始使用

请参考 [推理示例(Python)](../quick_start/python_demo) 和 [Python API 文档](../api_reference/python_api_index)。
