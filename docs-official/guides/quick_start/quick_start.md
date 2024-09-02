# 快速开始

## 一. 准备模型

Paddle Inference 原生支持由 [PaddlePaddle](https://github.com/PaddlePaddle/Paddle) 深度学习框架训练产出的推理模型。在Paddle 2.0后，模型保存为`model_name.pdmodel`和`model.pdiparams`两个文件，其中前者为模型结构信息，后者为模型权重信息，需要注意权重文件后辍需要为`.pdiparams`，如您的模型权重后辍为`.pdparams`表明该文件为训练过程中保存，需要导出为部署模型格式方可正常执行本文档的后续步骤。

- 如模型由各模型套件，如PaddleOCR/PaddleDetection/PaddleNLP等，请参考各套件的模型导出文档，导出部署模型
- 如模型是由飞桨框架API自行开发，请使用`paddle.jit.save`接口导出部署模型

如您的模型来源于其它深度学习框架，如PyTorch/TensorFlow/Caffe等，我们也提供了模型转换工具[X2Paddle](https://github.com/PaddlePaddle/X2Paddle.git),支持一键将外部模型转换为飞桨的模型格式，具体使用文档可参考[X2Paddle模型转换](https://github.com/PaddlePaddle/X2Paddle.git).

## 二、环境准备

Paddle Inference内置在Paddle框架中，但根据不同的部署需求，您可能需要下载不同的Paddle Inference预测包。 通常而言，对于Python部署，  
- 如您是直接通过`pip install paddlepaddle`，则已支持CPU（包括MKLDNN加速）部署；  
- 如您是直接通过`pip install paddlepaddle-gpu`安装，则已支持CPU(包括MKLDNN加速）、GPU部署

如您有其它部署需求，如TensorRT、ONNXRuntime后端的使用，Jetson环境部署，以及C++/C等其它编程语言部署，请在[Paddle Inference预编译预测库](../install/download_lib.md)页面自行下载, 或参考[源码编译](../install/compile/index_compile.md)页面编译安装。

## 二、模型部署

根据不同的编译语言，请分别参考

- 1. [预测示例(Python)](python_demo.md)
- 2. [预测示例(C++)](cpp_demo.md)
- 3. [一键动转静推理示例](jit_inference.md)
