# 环境与编译问题
如果您在配置环境或是编译过程中遇到错误，可以参考以下常见问题并尝试解决。


1. 编译报错, 且出错语句没有明显语法错误。  
**答：** 请检查使用的GCC版本，目前PaddlePaddle支持的编译器版本为 GCC 5.4 和 GCC 8.2.0。

2. 编译时报错`No CMAKE_CUDA_COMPILER could be found`。  
**答：** 编译时未找到 nvcc。设置编译选项 -DCMAKE_CUDA_COMPILER=nvcc 所在路径，注意需要与 CUDA 版本保持一致。