##########
安装指南
##########

本文介绍了 Paddle Inference 支持的硬件平台、操作系统环境、AI 软件加速库、多语言 API 等。  
  
安装 Paddle Inference 主要包括 **下载安装推理库** 和 **源码编译** 两种方式。  

 **下载安装推理库** 是最简单便捷的安装方式，Paddle Inference 提供了多种环境组合下的预编译库，如 cuda/cudnn 的多个版本组合、是否支持 TensorRT 、可选的 CPU 矩阵计算加速库等。详细内容可参考以下文档：

- 下载/安装推理库：

    - `Python安装 <python_install.html>`_ : 使用python wheel安装包直接安装  
    - `C++推理库 <cpp_install.html>`_ : 下载并使用C++预编译库  


如果用户环境与官网提供环境不一致（如用户环境的 cuda, cudnn, tensorrt 组合与预编译库提供的组合版本不一致），或对飞桨源代码有修改需求（如发现并修复了算子的 bug , 需要编译推理库集成测试），或希望进行定制化构建（如需新增算子、Pass 优化）等，则您可选择 **源码编译** 的方式。

- 源码编译：

    - `源码编译 <compile/index_compile.html>`_ : 在Linux、Windows、MacOS及其他平台上编译Paddle Inference


..  toctree::
    :hidden: 

    requirements.md
    python_install.md
    cpp_install.md
    download_lib.md
    compile/index_compile.rst
