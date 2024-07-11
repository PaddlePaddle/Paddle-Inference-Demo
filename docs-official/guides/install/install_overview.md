# Paddle Inference 安装概述

## 概述

本文主要介绍 Paddle Inference 安装概览，包括支持的硬件平台、操作系统环境、AI 软件加速库、多语言 API 等。

## 安装导航

安装 Paddle Inference 主要包括**下载推理库**和**源码编译**两种方式。

**下载推理库**是最简单便捷的安装方式，Paddle Inference 提供了多种环境组合下的预编译库，如 cuda/cudnn 的多个版本组合、是否支持 TensorRT 、可选的 CPU 矩阵计算加速库等。详细内容可参考 Paddle Inference [下载页面 TODO补充url]()。

如果用户环境与官网提供环境不一致（如用户环境的 cuda, cudnn, tensorrt 组合与预编译库提供的组合版本不一致），或对飞桨源代码有修改需求（如发现并修复了算子的 bug , 需要编译推理库集成测试），或希望进行定制化构建（如需新增算子、Pass 优化）等，则您可选择**源码编译**的方式。


1. 下载推理库请跳转到以下文档：

- [C++预编译库列表 TODO补充url]()
- [python whl包列表]()

2. 源码编译请参考以下文档：

- [CPU/GPU源码编译（Linux）TODO补充url，下同]()
- [CPU/GPU源码编译（Windows）]()
- [CPU/GPU源码编译（MacOS）]()
- [飞腾/鲲鹏源码编译（ARM）]()
- [申威源码编译（SW）]()
- [兆芯源码编译（x86）]()
- [龙芯源码编译（MIPS）]()

## 硬件平台

Paddle Inference 支持多种硬件平台，除了常规的 Intel CPU + NVIDIA GPU 组合外，还支持多种其它架构 CPU 和 AI 加速卡，如下列表所示。

CPU:
  1. X64: Intel(酷睿 Core, 志强 Xeon), AMD(Zen) 以及兆芯等；
  2. AArch64: 飞腾、鲲鹏；
  3. MIPS: 龙芯；
  4. SW: 申威。

AI 加速芯片:
  1. GPU: 主要指 NVIDIA(Kepler, Maxwell, Pascal, Volta, Turing, Ampere) 和 AMD 出产的 GPU;
  2. XPU: 昆仑加速卡；
  3. NPU: 昇腾加速卡；
  4. IPU: GraphCore加速卡。

## 操作系统

Paddle Inference 适配了多种操作系统，支持主流的 Windows, Mac, Linux，如下列表所示。

1. 主流 Linux 系统: Ubuntu, Centos, 统信 UOS, 银河麒麟 v10, 普华等
2. MacOS: 10.x/11.x (64 bit)
3. Windows10

## AI 软件加速库

Paddle Inference 为追求更快的性能，通过子图集成接入和算子接入两个方面，适配了以下 AI 软件加速库。

1. TensorRT: 以子图的方式接入；
2. cuDNN: 以算子的方式接入；
3. oneDNN: 以算子的方式接入；
4. Paddle Lite: 以子图的方式接入。

## 多语言 API

Paddle Inference 基于 C++ 实现，提供了标准 C++ API 接口，在此基础上封装了其它多语言 API，支持多语言 API 如下列表所示。

1. C++: 原生支持；
2. Python: 通过 `pybind` 进行封装；
