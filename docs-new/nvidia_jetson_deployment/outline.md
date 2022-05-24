# NV-GPU 推理概述

Paddle Infenrence 支持在 Ubuntu/Windows/MacOS 等操作系统上将模型部署在 GPU 上，同时也支持边缘端硬件，如Jetson NX、 Jetson Nano、 Jetson TX2。

Paddle Inference 支持 GPU 可通过两种方式来实现。（1）GPU 原生推理，PaddlePaddle 深度学习开源框架中存在大量用 CUDA 实现的算子，如果你选择用 GPU 原生推理，那么你训练好之后的模型会在一系列内部优化之后直接调用这些原生的算子实现 ；（2）GPU TensorRT 加速推理，将模型通过子图的方式接入 TensorRT，实现充分利用 GPU 的算力。

GPU 原生推理只支持 float32 精度的推理，float16 精度目前处于实验阶段。使用前，你需要确保你的机器上已经安装了 CUDA 和 cuDNN。

GPU TensorRT 加速推理需要借助于 NVIDIA TensorRT 软件库，TensorRT 是一个高性能的深度学习预测库，可为深度学习推理应用程序提供低延迟和高吞吐量。Paddle Inference 采用子图的方式对 TensorRT 进行了集成，即我们可以使用该模块来提升 Paddle Inference 的预测性能。TensorRT 接入方式支持 float32，float16，int8 精度的推理，除了 CUDA 和 cuDNN，使用前你还需确保你的机器上已经安装了 TensoRT 库。这通常会带给你比 GPU 原生推理更好的性能。


本章分为三个部分。

第一部分介绍如何使用 GPU 原生推理将模型部署在 GPU 硬件上，包括根据示例代码介绍 Paddle Inference C++/Python API 的使用流程，如何安装，以及如何在 Ubuntu、Windows等操作系统上编译和执行示例代码。

第二部分介绍如何使用 TensorRT 加速模型推理，根据示例代码介绍启用 TensorRT 加速的 API， 保存优化后的模型降低首帧耗时，如何支持动态shape等问题。还会介绍Paddle Inference 接入 TensorRT 的原理。

第三部分介绍 Paddle Inference 对量化或低精度推理的支持。
