#######
x86 CPU部署
#######

本章节介绍了如何使用Paddle Inference在x86 CPU平台上部署模型，请根据您的模型格式以及您期望的运行方式选择对应文档。

- `在x86 CPU上开发推理应用 <paddle_x86_cpu.html>`_: 原生CPU、MKLDNN和ONNX Runtime后端三种推理方式
- `在x86 CPU上部署BF16模型 <paddle_x86_cpu_bf16.html>`_: 部署BF16模型来提升推理性能
- `在x86 CPU上部署量化模型 <paddle_x86_cpu_int8.html>`_: 部署量化模型来提升推理性能

..  toctree::
    :hidden:
    
    paddle_x86_cpu.md
    paddle_x86_cpu_bf16.md
    paddle_x86_cpu_int8.md