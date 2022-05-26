# NV-GPU 推理概述

Paddle Inference 支持 GPU 可通过两种方式来实现。（1）GPU 原生推理，PaddlePaddle 深度学习开源框架中存在大量用 CUDA 实现的算子，如果您选择用 GPU 原生推理，那么用于部署的模型会在一系列内部优化之后直接调用这些原生算子实现 ；（2）GPU TensorRT 加速推理，Paddle Inference 将模型通过子图的方式接入 TensorRT，通过 TensorRT 来提升 Paddle Inference 在 GPU上 的推理性能。

GPU 原生推理仅支持 fp32，fp16 精度目前处于实验阶段。使用前，您需要确保您的机器上已经安装了 CUDA 和 cuDNN。

GPU TensorRT 加速推理需要借助于 TensorRT，TensorRT 是一个针对 NVIDIA GPU 及 Jetson 系列硬件的高性能机器学习推理 SDK，可以使得深度学习模型在这些硬件上的部署获得更好的性能。Paddle Inference 采用子图的方式对 TensorRT 进行了集成，即我们可以使用该模块来提升 Paddle Inference 的推理性能。TensorRT 接入方式支持 fp32，fp16，int8 精度的推理，除了 CUDA 和 cuDNN，使用前您还需确保您的机器上已经安装了 TensoRT 。这通常会带给您比 GPU 原生推理更好的性能。


本章分为三个部分。

第一部分介绍如何使用 GPU 原生推理将模型部署在 GPU 硬件上，包括根据示例代码介绍 Paddle Inference C++/Python API 的使用流程，如何安装 Paddle Inference 推理库，以及如何在 Ubuntu、Windows 等操作系统上编译和执行示例代码。

第二部分介绍如何使用 TensorRT 加速模型推理，根据示例代码介绍启用 TensorRT 加速的 API、 保存优化后的模型降低首帧耗时、支持动态 shape 的 API 等内容。还会介绍 Paddle Inference 接入 TensorRT 的原理。

第三部分介绍 Paddle Inference 对低精度和量化推理的支持。
