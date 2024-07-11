
# 系统要求

本篇文档介绍了安装Paddle Inference的软硬件要求，您可以根据以下文档描述，判断您的硬件平台、系统环境及软件依赖是否满足安装要求。

注： 在不满足要求的环境中安装 Paddle Inference 可能会遇到兼容性问题

## 硬件平台


Paddle Inference 支持多种硬件平台，除了常规的 Intel CPU + NVIDIA GPU 组合外，还支持多种其它架构 CPU 和 AI 加速卡，如下列表所示。

CPU:
1. X64: Intel(酷睿 Core, 志强 Xeon), AMD(Zen) 以及兆芯等；
2. AArch64: 飞腾、鲲鹏；
3. MIPS: 龙芯；
4. SW: 申威。

AI 加速芯片:
1. GPU: 主要指 NVIDIA(Kepler, Maxwell, Pascal, Volta, Turing, Ampere架构) 和 AMD 出产的 GPU;
2. XPU: 昆仑加速卡；
3. NPU: 昇腾加速卡；
4. IPU: GraphCore加速卡


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
