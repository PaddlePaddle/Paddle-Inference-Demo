# Paddle Inference 安装概述

## 概述

本文主要介绍 Paddle Inference 安装概览，包括支持的硬件平台、操作系统环境、AI 软件加速库、多语言 API 等。

## 硬件平台

Paddle Inference 支持多种硬件平台，除了常规的 INTEL CPU + NVIDIA GPU 组合外，还支持多种国产化 CPU 和 AI 加速卡，如下列表所示。

CPU:
  1. X64: intel、AMD以及兆芯；
  2. AArch64: 飞腾、鲲鹏；
  3. MIPS: 龙芯；
  4. SW: 申威。

AI 加速芯片:
  1. GPU: 主要指 NVIDIA 和 AMD 出产的 GPU;
  2. XPU: 昆仑加速卡；
  3. NPU: 昇腾加速卡；
  4. IPU: GraphCore加速卡。

## 操作系统

Paddle Inference 适配了多种操作系统，除了主流的 Windows, Mac, Linux 外，还支持多种国产化操作系统，如下列表所示。

1. 主流 Linux 系统
2. Mac
3. Windows10
4. 统信 UOS
5. 银河麒麟 v10
6. 普华

## AI 软件加速库

Paddle Inference 为追求更快的性能，通过子图集成接入和算子接入两个方面，适配了以下 AI 软件加速库。

1. TensorRT: 以子图的方式接入；
2. cuDNN: 以算子的方式接入；
3. oneDNN: 以算子的方式接入；
4. Paddle-Lite: 以子图的方式接入。

## 多语言 API

Paddle Inference 基于 C++ 实现，提供了标准 C++ API 接口，在此基础上封装了其它多语言 API，支持多语言 API 如下列表所示。

1. C++: 原生支持；
2. C: 通过 `extern "C"` 的方式进行封装；
3. Python: 通过 `pybind` 进行封装；
4. Go: 通过 `cgo` 在 CAPI 基础上进行封装。

## 安装导航

安装 Paddle Inference 主要包括**下载预测库**和**源码编译**两种方式。

**下载预测库**是最简单便捷的安装方式，Paddle Inference [下载页面 TODO补充url]()提供了多种环境组合下的预测库，用户可直接下载使用。

如果用户环境与官网提供环境不一致（如 cuda, cudnn, tensorrt 版本不一致等），或对飞桨源代码有修改需求，或希望进行定制化构建等，则您可选择**源码编译**的方式。


1. 下载预测库请跳转到以下文档：

- [C/C++预编译库列表 TODO补充url]()
- [python whl包列表]()
- [Go API 安装]()

2. 源码编译请参考以下文档：

- [CPU/GPU源码编译（Linux）TODO补充url，下同]()
- [CPU/GPU源码编译（Windows）]()
- [CPU/GPU源码编译（MacOS）]()
- [飞腾/鲲鹏源码编译（ARM）]()
- [申威源码编译（SW）]()
- [兆芯源码编译（x86）]()
- [龙芯源码编译（MIPS）]()
