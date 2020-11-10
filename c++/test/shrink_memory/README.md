## ShrinkMemory使用demo

### 一：获取MobilentV1测试模型

点击[链接](https://paddlepaddle-inference-banchmark.bj.bcebos.com/MobileNetV1_inference.tar)下载模型.

### 二：**样例编译**
 
文件`single_thread_test.cc` 为单线程使用ShrinkMemory降低内/显存的预测样例程序（程序中的输入为固定值，如果您有opencv或其他方式进行数据读取的需求，需要对程序进行一定的修改）。
文件`thread_local_test.cc` 为使用thread_local多线程使用ShrinkMemory降低内/显存的预测样例。
文件`multi_thread_test.cc` 为多线程使用ShrinkMemory降低内/显存的预测样例程序。
文件`CMakeLists.txt` 为编译构建文件。   
脚本`run_impl.sh` 包含了第三方库、预编译库的信息配置。

编译single_thread_test样例，我们首先需要对脚本`run_impl.sh` 文件中的配置进行修改。

1）**修改`run_impl.sh`**

打开`run_impl.sh`，我们对以下的几处信息进行修改：

```shell
# 根据需要选择single_thread_test, multi_thread_test, thread_local_test
DEMO_NAME=single_thread_test

# 根据预编译库中的version.txt信息判断是否将以下三个标记打开
WITH_MKL=ON
WITH_GPU=ON
USE_TENSORRT=OFF

# 配置预测库的根目录
LIB_DIR=${YOUR_LIB_DIR}/paddle_inference_install_dir

# 如果上述的WITH_GPU 或 USE_TENSORRT设为ON，请设置对应的CUDA， CUDNN， TENSORRT的路径。
CUDNN_LIB=/usr/local/cudnn/lib64
CUDA_LIB=/usr/local/cuda/lib64
# TENSORRT_ROOT=/usr/local/TensorRT-6.0.1.5
```

运行 `sh run_impl.sh`， 会在目录下产生build目录。


2） **运行样例**

```shell
# 进入build目录
cd build
# 运行样例
./build/single_thread_test -model_dir ${YOLO_MODEL_PATH} --use_gpu
# ./build/multi_thread_test --model_dir ${YOUR_MODEL_PATH} --use_gpu --thread_num 2
# ./build/thread_local_test --model_dir ${YOUR_MODEL_PATH} --use_gpu
```

运行过程中，请根据提示观测GPU的显存占用或CPU的内存占用，可以发现，当某次运行的batch_size很大时，会使得显/内存池较大，此时应用的显/内存占用较高，可以通过ShrinkMemory操作来显示的释放显/内存池。

### 更多链接
- [Paddle Inference使用Quick Start！]()
- [Paddle Inference Python Api使用]()
