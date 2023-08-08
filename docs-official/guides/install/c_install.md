# 安装 C API

本文主要介绍 Paddle Inference C API 的安装。主要分为以下三个章节：环境准备、安装步骤和验证安装。三个章节分别说明了安装前的环境要求、安装的具体流程和成功安装后的验证方法。

## 环境准备

- GCC 8.2+
- CMake 3.20+
- Visual Studio 2017 Update 3 （仅在使用 Windows 版本的推理库时需要，根据 Paddle 推理库所使用的 VS 版本选择，请参考 [Visual Studio 不同版本二进制兼容性](https://docs.microsoft.com/zh-cn/cpp/porting/binary-compat-2015-2017?view=msvc-170&viewFallbackFrom=vs-2019) )
- CUDA 10.2 / CUDA 11.2 / CUDA 11.6 / CUDA 11.7 / CUDA 11.8 / CUDA 12.0, cuDNN 7.6+, TensorRT （仅在使用 GPU 版本的推理库时需要）

您可参考 NVIDIA 官方文档了解 CUDA、cuDNN 和 TensorRT 的安装流程和配置方法，请见 [CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)，[cuDNN](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/)，[TensorRT](https://developer.nvidia.com/tensorrt)


Linux 下，版本对应关系如下表所示：

|CUDA 版本|cuDNN 版本| TensorRT 版本|
|---|---|---|
|10.2|7.6.5|7.0.0.11|
|10.2|8.1.1|7.0.0.11|
|11.2|8.2.1|8.0.3.4|
|11.6|8.4.0|8.4.0.6|
|11.7|8.4.1|8.4.2.4|
|11.8|8.6.0|8.5.1.7|
|12.0|8.9.1|8.6.1.6|

Windows 下，版本对应关系如下表所示：

|CUDA 版本|cuDNN 版本| TensorRT 版本|
|---|---|---|
|10.2|7.6.5|7.0.0.11|
|11.2|8.2.1|8.0.1.6|
|11.6|8.4.0|8.4.0.6|
|11.7|8.4.1|8.4.2.4|
|11.8|8.6.0|8.5.1.7|
|12.0|8.9.1|8.6.1.6|

## 开始安装

Paddle Inference 提供了 Linux/Windows/MacOS 平台的官方 Release 推理库下载，如果您使用的是以上平台，我们优先推荐您通过以下链接直接下载，或者您也可以参照文档进行[源码编译](./compile/index_compile.html)。

- [下载安装 Linux C 推理库](download_lib.html#id1)


- [下载安装 Windows C 推理库](download_lib.html#id3)


- [下载安装 MacOs C 推理库](download_lib.html#id6)

## 验证安装

### 静态验证方式

下载完成并解压之后，目录下的 `paddle_inference_c_install_dir` 即为 C 推理库，目录结构如下：

```
paddle_inference_c_install_dir
├── paddle
│   ├── include               C 推理库头文件目录
│   │   └── pd_common.h
│   │   └── pd_config.h
│   │   └── pd_inference_api.h         C 推理库头文件
│   │   └── pd_predictor.h
│   │   └── pd_tensor.h
│   │   └── pd_types.h
│   │   └── pd_utils.h
│   └── lib
│       ├── libpaddle_inference_c.a          C 静态推理库文件
│       └── libpaddle_inference_c.so         C 动态推理库文件
├── third_party
│   └── install                          第三方链接库和头文件
│       ├── cryptopp
│       ├── gflags
│       ├── glog
│       ├── mkldnn
│       ├── mklml
│       ├── protobuf
│       └── xxhash
└── version.txt                          版本信息与编译选项信息
```

include 目录下包括了使用飞桨推理库需要的头文件，lib 目录下包括了生成的静态库和动态库，third_party 目录下包括了推理库依赖的其它库文件。

其中 `version.txt` 文件中记录了该推理库的版本信息，包括 Git Commit ID、使用 OpenBLAS 或 MKL 数学库、CUDA/cuDNN 版本号，如：

```bash
GIT COMMIT ID: 47fa64004362b1d7d63048016911e62dc1d84f45
WITH_MKL: ON
WITH_MKLDNN: ON
WITH_GPU: ON
WITH_ROCM: OFF
WITH_ASCEND_CL: OFF
WITH_ASCEND_CXX11: OFF
WITH_IPU: OFF
CUDA version: 11.2
CUDNN version: v8.2
CXX compiler version: 8.2.0
WITH_TENSORRT: ON
TensorRT version: v8.2.4.2
```

### 动态验证方式

您可以编写应用代码，与推理库联合编译并测试结果。请参考 [推理示例(C)](../quick_start/c_demo) 一节。

## 开始使用

请参考 [推理示例(C)](../quick_start/c_demo) 和 [C API 文档](../../api_reference/c_api_doc/c_api_index)。
