## 自定义算子模型构建运行示例

### 一：获取本样例中的自定义算子模型
待上传。

### 二：**样例编译**

文件 `custom_relu_op.cc`、`custom_relu_op.cu`、`custom_relu_op_dup.cc` 为自定义算子源文件，自定义算子编写方式请参考[飞桨官网文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/index_cn.html)。
注意：自定义算子目前是试验功能，需要依赖 boost，并需要与飞桨预测库 `libpaddle_inference.so` 联合构建。

文件`custom_op_test.cc` 为预测的样例程序。    
文件`CMakeLists.txt` 为编译构建文件。   
脚本`run_impl.sh` 包含了第三方库、预编译库的信息配置。

我们首先需要对脚本`run_impl.sh` 文件中的配置进行修改。

1）**修改`run_impl.sh`**

打开`run_impl.sh`，我们对以下的几处信息进行修改：

```shell
# 根据预编译库中的version.txt信息判断是否将以下三个标记打开
WITH_MKL=ON       
WITH_GPU=ON         
USE_TENSORRT=OFF

# 配置预测库的根目录
LIB_DIR=${YOUR_LIB_DIR}/paddle_inference_install_dir

# 如果上述的WITH_GPU 或 USE_TENSORRT设为ON，请设置对应的CUDA， CUDNN， TENSORRT的路径。
CUDNN_LIB=/paddle/nvidia-downloads/cudnn_v7.5_cuda10.1/lib64
CUDA_LIB=/paddle/nvidia-downloads/cuda-10.1/lib64
# TENSORRT_ROOT=/paddle/nvidia-downloads/TensorRT-6.0.1.5
```

运行 `sh run_impl.sh`， 会在目录下产生build目录。


2） **运行样例**

```shell
# 进入build目录
cd build
# 运行样例
./custom_op_test
```

运行结束后，程序会将模型结果打印到屏幕，说明运行成功。

### 更多链接
- [Paddle Inference使用Quick Start！](https://paddle-inference.readthedocs.io/en/latest/introduction/quick_start.html)
- [Paddle Inference C++ Api使用](https://paddle-inference.readthedocs.io/en/latest/api_reference/cxx_api_index.html)
- [Paddle Inference Python Api使用](https://paddle-inference.readthedocs.io/en/latest/api_reference/python_api_index.html)
