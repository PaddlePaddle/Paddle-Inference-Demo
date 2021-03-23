## 运行C++ Ernie变长输入样例

### 一：获取Ernie模型

点击[链接](http://paddle-inference-dist.bj.bcebos.com/tensorrt_test/ernie_model_4.tar.gz)下载模型， 如果你想获取更多的**Ernie模型信息**，请访问[这里](https://www.paddlepaddle.org.cn/paddle/ernie)。
当前Paddle Inference支持ernie的以下两种输入方式，变长输入性能更佳，本示例是变长输入的使用示例，输入方式一可以参考[单测代码](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/inference/tests/api/trt_dynamic_shape_ernie_test.cc)。 

1）动态shape，单batch内padding到固定长度。支持FP32和FP16精度。   
batch 1， shape {3, 4, 1}，输入数据如下，其中'X'为padding数据
```
aaaX
bbbb
cXXX
```
batch 2， shape {2, 5, 1}，输入数据如下，其中'X'为padding数据
```
eeeXX
fffff
```

2）动态shape，同时batch内数据支持变长，不用padding数据。当前只支持FP16精度。  
batch1
```
Data: aaabbbbc
ShapeData: 0, 3, 7, 8
```
batch2
```
Data: eeefffff
ShapeData: 0, 3, 8
```

### 二：样例编译

文件`ernie_varlen_test.cc` 为预测的样例程序（程序中的输入为固定值，如果您有opencv或其他方式进行数据读取的需求，需要对程序进行一定的修改）。
脚本`compile.sh` 包含了第三方库、预编译库的信息配置。

1）**修改`compile.sh`**

打开`compile.sh`，我们对以下的几处信息进行修改：

```shell
# 根据预编译库中的version.txt信息判断是否将以下三个标记打开
WITH_MKL=ON
WITH_GPU=ON
USE_TENSORRT=ON

# 配置预测库的根目录
LIB_DIR=${work_path}/../lib/paddle_inference

# 如果上述的WITH_GPU 或 USE_TENSORRT设为ON，请设置对应的CUDA， CUDNN， TENSORRT的路径。
CUDNN_LIB=/usr/lib/x86_64-linux-gnu/
CUDA_LIB=/usr/local/cuda/lib64
TENSORRT_ROOT=/root/work/nvidia/TensorRT-6.0.1.5.cuda-10.1.cudnn7.6-OSS7.2.1
```
TIPS:Ernie变长输入需要TensorRT7.2.1+或者低版本的TensorRT联合编译OSS 7.2.1（[TensorRT Open Source Software 7.2.1](https://github.com/NVIDIA/TensorRT/tree/7.2.1) ）。

运行 `bash compile.sh`， 会在目录下产生build目录。

2） **运行样例**

```shell
# 运行样例
./build/ernie_varlen_test --model_dir=/your/downloaded/model/path/here
```

运行结束后，程序会将模型结果打印到屏幕，说明运行成功。


### 更多链接
- [Paddle Inference使用Quick Start！](https://paddle-inference.readthedocs.io/en/latest/introduction/quick_start.html)
- [Paddle Inference C++ Api使用](https://paddle-inference.readthedocs.io/en/latest/api_reference/cxx_api_index.html)
- [Paddle Inference Python Api使用](https://paddle-inference.readthedocs.io/en/latest/api_reference/python_api_index.html)
