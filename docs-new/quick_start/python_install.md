# Python 推理部署

本文主要介绍 Paddle Inference Python API 的安装。主要分为以下三个章节：环境准备、安装步骤和验证安装。三个章节分别说明了安装前的环境要求、安装的具体流程和成功安装后的验证方法。

## 环境准备

- Python: 3.6 / 3.7 / 3.8 / 3.9
- CUDA 10.1 / CUDA 10.2 / CUDA 11.0 / CUDA 11.2, cuDNN7.6+, tensorrt （仅在使用 gpu 版本的预测库时需要）

（您可参考 nvidia 官方文档了解 CUDA 和 cuDNN 的安装流程和配置方法，请见 [CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)，[cuDNN](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/)，版本对应关系如下表所示)

|CUDA 版本|cuDNN 版本| TensorRT 版本|
|---|---|---|
|10.1|7.6|6|
|10.2|7.6|7|
|11.0|8.0|7|
|11.2|8.2|8|

## 开始安装

请参照 [官方主页-开始安装](https://www.paddlepaddle.org.cn/install/quick) 页面进行自行安装或编译，当前支持 pip/conda 安装，docker 镜像以及[源码编译](../user_guides/source_compile)等多种方式来准备 Paddle Inference 开发环境。

### 通过 pip 在线安装（推荐）

- CPU版本
```
python3 -m pip install paddlepaddle==2.3.0 -i https://mirror.baidu.com/pypi/simple
```
- GPU版本（以 CUDA11.0 为例）
```
python3 -m pip install paddlepaddle-gpu==2.3.0.post110 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
```

### 您也可以下载 whl 包到本地，然后通过 pip 工具安装

- [下载安装 Linux Python installer](../user_guides/download_lib.html#python)
- [下载安装 Windows Python installer](../user_guides/download_lib.html#id4)


## 验证安装

安装完成后，可以使用 python3 进入python解释器，输入以下指令，出现 `Your Paddle Fluid is installed successfully! ` ，说明安装成功。

```python
import paddle.fluid as fluid
fluid.install_check.run_check()
```

## 开始使用

请参考 [预测示例(Python)](../quick_start/python_demo) 和 [Python API 文档](../api_reference/python_api_index)。
