# C++API安装

本文主要介绍 Paddle Inference C++ API 的安装。主要分为以下三个章节：环境准备、安装步骤和验证安装。三个章节分别说明了安装前的环境要求、安装的具体流程和成功安装后的验证方法。

## 环境准备

- G++ 8.2
- CMake 3.0+
- Visual Studio 2017 Update 3 （仅在使用 Windows 版本的预测库时需要，根据 Paddle 预测库所使用的 VS 版本选择，请参考 [Visual Studio 不同版本二进制兼容性](https://docs.microsoft.com/zh-cn/cpp/porting/binary-compat-2015-2017?view=msvc-170&viewFallbackFrom=vs-2019) )
- cuda 10.1 / cuda 10.2 / cuda 11.0 / cuda 11.2, cuDNN7.6+, tensorrt （仅在使用 gpu 版本的预测库时需要）

（您可参考 nvidia 官方文档了解 CUDA 和 cuDNN 的安装流程和配置方法，请见 [cuda](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)，[cuDNN](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/)，版本对应关系如下表所示)

|CUDA 版本|cuDNN 版本| TensorRT 版本|
|---|---|---|
|10.1|7.6|6|
|10.2|7.6|7|
|11.0|8.0|7|
|11.2|8.2|8|

## 开始安装

Paddle Inference 提供了 Linux/Windows/MacOS 平台的官方 Release 预测库下载，如果您使用的是以上平台，我们优先推荐您通过以下链接直接下载，或者您也可以参照文档进行[源码编译](compile/index_compile.html)。

- [下载安装 Linux 预测库](download_lib.html#linux)

- [下载安装 Windows 预测库](download_lib.html#windows)

- [下载安装 MacOs预测库](download_lib.html#mac)

## 验证安装

### 静态验证方式

下载完成并解压之后，目录下的 `paddle_inference_install_dir` 即为 C++ 预测库，目录结构如下：

```bash
paddle_inference/paddle_inference_install_dir/
├── CMakeCache.txt
├── paddle
│   ├── include                                    C++ 预测库头文件目录
│   │   ├── crypto
│   │   ├── internal
│   │   ├── paddle_analysis_config.h
│   │   ├── paddle_api.h
│   │   ├── paddle_infer_declare.h
│   │   ├── paddle_inference_api.h                 C++ 预测库头文件
│   │   ├── paddle_mkldnn_quantizer_config.h
│   │   └── paddle_pass_builder.h
│   └── lib
│       ├── libpaddle_inference.a                      C++ 静态预测库文件
│       └── libpaddle_inference.so                     C++ 动态态预测库文件
├── third_party
│   ├── install                                    第三方链接库和头文件
│   │   ├── cryptopp
│   │   ├── gflags
│   │   ├── glog
│   │   ├── mkldnn
│   │   ├── mklml
│   │   ├── protobuf
│   │   └── xxhash
│   └── threadpool
│       └── ThreadPool.h
└── version.txt
```

include 目录下包括了使用飞桨预测库需要的头文件，lib 目录下包括了生成的静态库和动态库，third_party 目录下包括了预测库依赖的其它库文件。

其中 `version.txt` 文件中记录了该预测库的版本信息，包括 Git Commit ID、使用 OpenBlas 或 MKL 数学库、CUDA/cuDNN 版本号，如：

```bash
GIT COMMIT ID: 1bf4836580951b6fd50495339a7a75b77bf539f6
WITH_MKL: ON
WITH_MKLDNN: ON
WITH_GPU: ON
CUDA version: 9.0
cuDNN version: v7.6
CXX compiler version: 4.8.5
WITH_TENSORRT: ON
TensorRT version: v6
```

### 动态验证方式

您可以编写应用代码，与预测库联合编译并测试结果。请参考 [预测示例(C++)](../quick_start/cpp_demo) 一节。

## 开始使用

请参考 [预测示例(C++)](../quick_start/cpp_demo) 和 [C++ API 文档](../api_reference/cpp_api_index)。
