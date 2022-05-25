# 自定义算子模型构建运行示例

## 一：获取本样例中的自定义算子模型
下载地址：https://paddle-inference-dist.bj.bcebos.com/inference_demo/custom_operator/custom_relu_infer_model.tgz

执行 `tar zxvf custom_relu_infer_model.tgz` 将模型文件解压至当前目录。

## 二：**样例编译**

文件 `custom_relu_op.cc`、`custom_relu_op.cu` 为自定义算子源文件，自定义算子编写方式请参考[飞桨官网文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/index_cn.html)。
注意：自定义算子目前需要与飞桨预测库 `libpaddle_inference.so` 联合构建。

文件`custom_op_test.cc` 为预测的样例程序。
文件`CMakeLists.txt` 为编译构建文件。
脚本`compile.sh` 包含了第三方库、预编译库的信息配置。

我们首先需要对脚本`compile.sh` 文件中的配置进行修改。

1）**修改`compile.sh`**

打开`compile.sh`，我们对以下的几处信息进行修改：

```shell
# 根据预编译库中的version.txt信息判断是否将以下三个标记打开
WITH_MKL=ON
WITH_GPU=ON
USE_TENSORRT=OFF

# 配置预测库的根目录
LIB_DIR=${work_path}/../lib/paddle_inference

# 如果上述的WITH_GPU 或 USE_TENSORRT设为ON，请设置对应的CUDA， CUDNN， TENSORRT的路径。
CUDNN_LIB=/usr/lib/x86_64-linux-gnu/
CUDA_LIB=/usr/local/cuda/lib64
TENSORRT_ROOT=/usr/local/TensorRT-7.0.0.11
```

运行 `bash compile.sh`， 会在目录下产生build目录。


2） **运行样例**

```shell
# 运行样例
./build/custom_op_test
```

运行结束后，程序会将模型结果打印到屏幕，说明运行成功。

> 注：确保路径配置正确后，也可执行执行 `bash run.sh` ，一次性完成以上两个步骤的执行

## 更多链接
- [Paddle Inference使用Quick Start！](https://paddle-inference.readthedocs.io/en/latest/introduction/quick_start.html)
- [Paddle Inference C++ Api使用](https://paddle-inference.readthedocs.io/en/latest/api_reference/cxx_api_index.html)
- [Paddle Inference Python Api使用](https://paddle-inference.readthedocs.io/en/latest/api_reference/python_api_index.html)
