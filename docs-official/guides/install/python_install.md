# 安装 Python API

本文主要介绍 Paddle Inference Python API 的安装。主要分为以下三个章节：环境准备、安装步骤和验证安装。三个章节分别说明了安装前的环境要求、安装的具体流程和成功安装后的验证方法。

## 环境准备

- Python: 3.6 / 3.7 / 3.8 / 3.9
- Python: 3.6 / 3.7 / 3.8 / 3.9 / 3.10
- CUDA 10.1 / CUDA 10.2 / CUDA 11.1 / CUDA 11.2 / CUDA 11.6 / CUDA 11.7, cuDNN7.6+, TensorRT （仅在使用 GPU 版本的推理库时需要）

您可参考 NVIDIA 官方文档了解 CUDA 和 cuDNN 的安装流程和配置方法，请见 [CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)，[cuDNN](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/)，版本对应关系如下表所示：

您可参考 NVIDIA 官方文档了解 CUDA、cuDNN 和 TensorRT 的安装流程和配置方法，请见 [CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)，[cuDNN](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/)，[TensorRT](https://developer.nvidia.com/tensorrt)


Linux 下，版本对应关系如下表所示：

|CUDA 版本|cuDNN 版本| TensorRT 版本|
|---|---|---|
|10.1|7.6.5||
|10.2|7.6.5|7.0.0.11|
|11.1|8.1.1|7.2.3.4|
|11.2|8.1.1|8.0.3.4|
|11.2|8.2.1|8.0.3.4|
|11.6|8.4.0|8.4.0.6|
|11.7|8.4.1|8.4.2.4|

Windows 下，版本对应关系如下表所示：

|CUDA 版本|cuDNN 版本| TensorRT 版本|
|---|---|---|
|10.1|7.6.5||
|10.2|7.6.5|7.0.0.11|
|11.1|8.1.1|8.0.3.4|
|11.2|8.2.1|8.2.4.2|
|11.6|8.4.0|8.4.0.6|
|11.7|8.4.1|8.4.2.4|

## 开始安装

### 方式一：通过 pip 在线安装（包含 TensorRT）

参考[Pip 安装](https://www.paddlepaddle.org.cn/documentation/docs/zh/2.4rc/install/pip/frompip.html)

### 方式二：下载 whl 包（可选 TensorRT）到本地，然后通过 pip 工具安装

[下载安装 Linux Python installer](download_lib.html#python)

Linux 下仅提供 CUDA11.2/cuDNN8.2.1/TensorRT8.0.3.4 版本组合的 whl 包，其他组合请使用方式一(pip 在线安装)

### 方式三：源码安装

参考[源码编译](./compile/index_compile.html)文档。

## 验证安装

### 静态验证方式

安装完成后，可以使用 python3 进入 python 解释器，输入以下指令，出现 `PaddlePaddle is installed successfully! ` ，说明安装成功。

```python
import paddle
paddle.utils.run_check()
```

### 动态验证方式

您可以编写应用代码并测试结果。请参考 [推理示例(Python)](../quick_start/python_demo) 一节。


## 开始使用

请参考 [推理示例(Python)](../quick_start/python_demo) 和 [Python API 文档](../api_reference/python_api_index)。
