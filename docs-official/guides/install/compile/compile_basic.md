# 源码编译基础

## 什么时候需要源码编译？

深度学习的发展十分迅速，对科研或工程人员来说，可能会遇到一些需要自己开发 OP 的场景，可以在 Python 层面编写 OP，但如果对性能有严格要求的话则必须在 C++ 层面开发 OP，对于这种情况，需要用户源码编译飞桨，使之生效。

此外对于绝大多数使用 C++ 将模型部署上线的工程人员来说，您可以直接通过飞桨官网下载已编译好的推理库，快捷开启飞桨使用之旅。[飞桨官网](https://www.paddlepaddle.org.cn/documentation/docs/zh/advanced_guide/inference_deployment/inference/build_and_install_lib_cn.html) 提供了多个不同环境下编译好的推理库。如果用户环境与官网提供环境不一致（如 CUDA、 cuDNN、 TensorRT 版本不一致等），或对飞桨源代码有修改需求，或希望进行定制化构建，可查阅本文档自行源码编译得到推理库。

## 目标产物

飞桨框架的源码编译包括源代码的编译和链接，最终生成的目标产物包括：C++ lib 和 Python whl包。

**c++ lib**

含有 C++ 接口的头文件及其二进制库：用于 C++ 环境，将文件放到指定路径即可开启飞桨使用之旅。
推理库编译后，所有产出均位于 build 目录下的 paddle_inference_install_dir 目录内，目录结构如下。version.txt 中记录了该推理库的版本信息，包括 Git Commit ID、使用 OpenBlas 或 MKL 数学库、CUDA/cuDNN 版本号。

```shell
build/paddle_inference_install_dir
├── CMakeCache.txt
├── paddle
│   ├── include
│   │   ├── paddle_anakin_config.h
│   │   ├── paddle_analysis_config.h
│   │   ├── paddle_api.h
│   │   ├── paddle_inference_api.h
│   │   ├── paddle_mkldnn_quantizer_config.h
│   │   └── paddle_pass_builder.h
│   └── lib
│       ├── libpaddle_inference.a (Linux)
│       ├── libpaddle_inference.so (Linux)
│       └── libpaddle_inference.lib (Windows)
├── third_party
│   ├── boost
│   │   └── boost
│   ├── eigen3
│   │   ├── Eigen
│   │   └── unsupported
│   └── install
│       ├── gflags
│       ├── glog
│       ├── mkldnn
│       ├── mklml
│       ├── protobuf
│       ├── xxhash
│       └── zlib
└── version.txt
```

Include 目录下包括了使用飞桨推理库需要的头文件，lib 目录下包括了生成的静态库和动态库，third_party 目录下包括了推理库依赖的其它库文件。

您可以编写应用代码，与推理库联合编译并测试结果。请参考 [C++ 推理库 API 使用](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/05_inference_deployment/inference/native_infer.html) 一节。

**python whl 包**

Python Wheel 形式的安装包：用于 Python 环境，也就是说，通过 pip 安装属于在线安装，这里属于本地安装。
编译完毕后，会在 python/dist 目录下生成一个 Python Wheel 安装包，安装测试的命令为：

```shell
pip3 install [wheel 包的名字]
```

安装完成后，可以使用 python3 进入 python 解释器，输入以下指令，出现 `PaddlePaddle is installed successfully! ` ，说明安装成功。

```shell
import paddle
paddle.utils.run_check()
```

## 基础概念

飞桨深度学习框架主要由 C++ 语言编写，通过 pybind 工具提供了 Python 端的接口，飞桨的源码编译主要包括编译和链接两步。

编译过程由编译器完成，编译器以编译单元（后缀名为 .cc 或 .cpp 的文本文件）为单位，将 C++ 语言 ASCII 源代码翻译为二进制形式的目标文件。一个工程通常由若干源码文件组织得到，所以编译完成后，将生成一组目标文件。

链接过程使分离编译成为可能，由链接器完成。链接器按一定规则将分离的目标文件组合成一个能映射到内存的二进制程序文件，并解析引用。由于这个二进制文件通常包含源码中指定可被外部用户复用的函数接口，所以也被称作函数库。根据链接规则不同，链接可分为静态和动态链接。静态链接对目标文件进行归档；动态链接使用地址无关技术，将链接放到程序加载时进行。

配合包含类、函数等声明的头文件（后缀名为 .h 或 .hpp），用户可以复用程序库中的代码开发应用。静态链接构建的应用程序可独立运行，而动态链接程序在加载运行时需到指定路径下搜寻其依赖的二进制库。

## 编译方式

飞桨框架的设计原则之一是满足不同平台的可用性。然而，不同操作系统惯用的编译和链接器是不一样的，使用它们的命令也不一致。比如，Linux 一般使用 GNU 编译器套件（GCC），Windows 则使用 Microsoft Visual C++（MSVC）。为了统一编译脚本，飞桨使用了支持跨平台构建的 CMake，它可以输出上述编译器所需的各种 Makefile 或者 Project 文件。

为方便编译，框架对常用的 CMake 命令进行了封装，如仿照 Bazel 工具封装了 cc_binary 和 cc_library ，分别用于可执行文件和库文件的产出等，对 CMake 感兴趣的同学可在 cmake/generic.cmake 中查看具体的实现逻辑。Paddle 的 CMake 中集成了生成 python wheel 包的逻辑，对如何生成 wheel 包感兴趣的同学可参考 [相关文档](https://packaging.python.org/tutorials/packaging-projects/)。


## 编译步骤

飞桨分为 CPU 版本和 GPU 版本。如果您的计算机没有 Nvidia GPU，请选择 CPU 版本构建安装。如果您的计算机含有 Nvidia GPU 且预装有 CUDA / cuDNN，也可选择 GPU 版本构建安装。下面提供下在不同平台上的编译步骤：

[Linux 下从源码编译](source_compile_under_Linux.md)

[Windows 下从源码编译](source_compile_under_Windows.md)

[MacOs 下从源码编译](source_compile_under_MacOS.md)
