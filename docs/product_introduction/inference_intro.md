# Paddle Inference 简介

Paddle Inference 是飞桨的原生推理库， 作用于服务器端和云端，提供高性能的推理能力。

由于能力直接基于飞桨的训练算子，因此Paddle Inference 可以通用支持飞桨训练出的所有模型。

Paddle Inference 功能特性丰富，性能优异，针对不同平台不同的应用场景进行了深度的适配优化，做到高吞吐、低时延，保证了飞桨模型在服务器端即训即用，快速部署。

## Paddle Inference的高性能实现

### 内存/显存复用提升服务吞吐量

在推理初始化阶段，对模型中的OP输出Tensor 进行依赖分析，将两两互不依赖的Tensor在内存/显存空间上进行复用，进而增大计算并行量，提升服务吞吐量。

### 细粒度OP横向纵向融合减少计算量

在推理初始化阶段，按照已有的融合模式将模型中的多个OP融合成一个OP，减少了模型的计算量的同时，也减少了 Kernel Launch的次数，从而能提升推理性能。目前Paddle Inference支持的融合模式多达几十个。

### 内置高性能的CPU/GPU Kernel

内置同Intel、Nvidia共同打造的高性能kernel，保证了模型推理高性能的执行。

### 子图集成TensorRT加快GPU推理速度

Paddle Inference采用子图的形式集成TensorRT，针对GPU推理场景，TensorRT可对一些子图进行优化，包括OP的横向和纵向融合，过滤冗余的OP，并为OP自动选择最优的kernel，加快推理速度。

### 子图集成Paddle Lite轻量化推理引擎

Paddle Lite 是飞桨深度学习框架的一款轻量级、低框架开销的推理引擎，除了在移动端应用外，还可以使用服务器进行 Paddle Lite 推理。Paddle Inference采用子图的形式集成 Paddle Lite，以方便用户在服务器推理原有方式上稍加改动，即可开启 Paddle Lite 的推理能力，得到更快的推理速度。并且，使用 Paddle Lite 可支持在百度昆仑等高性能AI芯片上执行推理计算。

### 支持加载PaddleSlim量化压缩后的模型

PaddleSlim是飞桨深度学习模型压缩工具，Paddle Inference可联动PaddleSlim，支持加载量化、裁剪和蒸馏后的模型并部署，由此减小模型存储空间、减少计算占用内存、加快模型推理速度。其中在模型量化方面，Paddle Inference在X86 CPU上做了深度优化，常见分类模型的单线程性能可提升近3倍，ERNIE模型的单线程性能可提升2.68倍。

## Paddle Inference的通用性

### 主流软硬件环境兼容适配

支持服务器端X86 CPU、NVIDIA GPU芯片，兼容Linux/Mac/Windows系统。支持所有飞桨训练产出的模型，完全做到即训即用。

### 多语言环境丰富接口可灵活调用

支持C++, Python, C，接口简单灵活，20行代码即可完成部署。对于其他语言，提供了ABI稳定的C API, 用户可以很方便地扩展。

