源码编译
##########

对于首次接触Paddle Inference源码编译， 或者不确定自己是否需要从源码编译的用户，我们建议您首先阅读 **源码编译基础** 进行了解。

- `源码编译基础 <compile_basic.html>`_  

如果您确定需要进行源码编译，请首先确定您的硬件平台：

 **(1) x86 CPU 或 Nvidia GPU**

如果您的计算机没有 Nvidia GPU，请选择 CPU 版本构建安装。如果您的计算机含有 Nvidia GPU 且预装有 CUDA / cuDNN，也可选择 GPU 版本构建安装。下面提供下在不同OS平台上的编译步骤：

-  `Linux 下从源码编译 <source_compile_under_Linux.html>`_ 

-  `Windows 下从源码编译 <source_compile_under_Windows.html>`_ 

-  `MacOs 下从源码编译 <source_compile_under_MacOS.html>`_ 

 **（2）其他硬件环境编译**

如果您的硬件环境不同于以上介绍的通用环境，您可以前往 `其他硬件部署 <../../hardware_support/index_hardware.html>`_ 页面查阅您的硬件平台是否被Paddle Inference支持，并参考相应文档完成编译和推理应用开发。


..  toctree::
    :hidden:
    
    compile_basic.md
    source_compile_under_Linux.md
    source_compile_under_Windows.md
    source_compile_under_MacOS.md

