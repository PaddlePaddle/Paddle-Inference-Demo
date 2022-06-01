# 运行报错
如果您的推理应用程序在运行中出现报错提示或异常退出，可以参考以下常见问题并尝试解决。


1. 运行时报错`RuntimeError: parallel_for failed: cudaErrorNoKernelImageForDevice: no kernel image is available for execution on the device`。  
**答：** 这种情况一般出现在编译和运行在不同架构的显卡上，且cmake时未指定运行时需要的CUDA架构。可以在cmake时加上 -DCUDA_ARCH_NAME=All（或者特定的架构如Turing、Volta、Pascal等），否则会使用默认值Auto，此时只会当前的CUDA架构编译。

2. 运行时报错`PaddleCheckError: Expected id < GetCUDADeviceCount(), but received id:0 >= GetCUDADeviceCount():0` 。  
**答：** 一般原因是找不到驱动文件。请检查环境设置，需要在LD_LIBRARY_PATH中加入/usr/lib64（驱动程序libcuda.so所在的实际路径）。

3. 运行时报错`Error: Cannot load cudnn shared library.` 。  
**答：** 请在LD_LIBRARY_PATH中加入cuDNN库的路径。

4. 运行时报错libstdc++.so.6中GLIBCXX版本过低。   
**答：** 运行时链接了错误的glibc。可以通过以下两种方式解决：  
1）在编译时，通过在CXXFLAGS中加入"-Wl,--rpath=/opt/compiler/gcc-8.2/lib,--dynamic-linker=/opt/compiler/gcc-8.2/lib/ld-linux-x86-64.so.2"，来保证编译产出的可执行程序链接到正确的libc.so.6/libstdc++.so.6库。  
2）在运行时，将正确的glibc的路径加入到LD_LIBRARY_PATH中。

5. 运行时报错`CUDNN_STATUS_NOT_INITIALIZED at ...`。  
**答：** 请检查运行时链接的cuDNN版本和编译预测库使用的cuDNN版本是否一致。

6. 运行时报错`OMP: Error #100 Fatal system error detected`。
**答：** OMP的问题，可参考链接[ https://unix.stackexchange.com/questions/302683/omp-error-bash-on-ubuntu-on-windows]( https://unix.stackexchange.com/questions/302683/omp-error-bash-on-ubuntu-on-windows)

7. 初始化阶段报错`Program terminated with signal SIGILL, Illegal instruction`。  
**答：** 请检查下是否在不支持AVX的机器上使用了AVX的预测库。

8. 在JetPack 4.4环境的Jetson开发套件上运行带卷积的模型报错`terminate called after throwing an instance of 'std::logic_error' what(): basic_string::_M_construct null not valid`。  
**答：** 这个是cuDNN 8.0在SM_72下的bug，在运行cudnnConvolutionBiasActivationForward的时候会出错，见[https://forums.developer.nvidia.com/t/nx-jp44-cudnn-internal-logic-error/124805](https://forums.developer.nvidia.com/t/nx-jp44-cudnn-internal-logic-error/124805)。  
目前有以下两种解决方案：  
1）通过`config.pass_builder()->DeletPass()`删除如下PASS：`conv_elementwise_add_act_fuse_pass`、`conv_elementwise_add2_act_fuse_pass`、`conv_elementwise_add_fuse_pass`, 来避免预测期间进行conv + bias + activation的融合。  
2）将cuDNN 8.0降级为cuDNN 7.6。

9. 运行时报错`Expected static_cast<size_t>(col) < feed_list.size(), but received static_cast<size_t>(col):23 >= feed_list.size():0`。  
**答：** 在2.0 rc1之前的版本，用户使用ZeroCopyTensor和ZeroCopyRun接口时，需要设置`config.SwitchUseFeedFetchOps(false)`，后续版本已经隐藏ZeroCopyTensor的设计，无需手动设置。 

10. 运行时报错`PaddleCheckError: an illegal memory access was encounted at xxx`。  
**答：** 请检查输入Tensor 是否存在指针越界。  