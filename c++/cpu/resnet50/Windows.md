# Windows 运行 ResNet50 图像分类样例

ResNet50 样例展示了单输入模型在 CPU 下使用 oneDNN 和 OnnxRuntime 的推理过程。本文档基于`Visual Studio 2019`编写，最好使用相同版本运行样例。运行步骤如下：

## 一：获取 Paddle Inference 预测库

- [官网下载](https://www.paddlepaddle.org.cn/documentation/docs/zh/advanced_guide/inference_deployment/inference/build_and_install_lib_cn.html)
- 自行编译获取

将获取到的 Paddle Inference 预测库放在本目录中，后面编译时需要填写预测库路径。

## 二：获取 Resnet50 模型

点击[链接](https://paddle-inference-dist.bj.bcebos.com/Paddle-Inference-Demo/resnet50.tgz)下载模型。如果你想获取更多的**模型训练信息**，请访问[这里](https://github.com/PaddlePaddle/PaddleClas)。

## 三：编译样例

文件`resnet50_test.cc` 为预测的样例程序（程序中的输入为固定值，如果您有 opencv 或其他方式进行数据读取的需求，需要对程序进行一定的修改）

### 1. 编译准备
需要先将`Paddle-Inference-Demo/c++/lib`下的`CMakeLists.txt`拷贝至本目录

### 2.使用命令编译
开始菜单栏找到`Visual Studio 2019`目录下的`x64 Native Tools Command Prompt for VS 2019`并打开 使用下面的`cmake`命令生成vs工程
```
# 进入build目录
mkdir build && cd build

# 执行cmake指令, 其中 PADDLE_LIB 的路径要改为预测库所在的路径
cmake .. -G "Visual Studio 16 2019" -A x64 -DCMAKE_BUILD_TYPE=Release -DWITH_MKL=ON -DDEMO_NAME=resnet50_test -DWITH_ONNXRUNTIME=ON -DWITH_STATIC_LIB=OFF -DPADDLE_LIB=path/to/paddle_inference

# 执行编译
msbuild cpp_inference_demo.sln /m /p:Configuration=Release /p:Platform=x64
```
执行完之后在`build`目录下出现Release目录，目录中的`resnet50_test.exe`就是编译出来的可执行文件。


## 四：运行样例

### 使用 oneDNN 运行样例
```shell
.\Release\resnet50_test --model_file ..\resnet50\inference.pdmodel --params_file ..\resnet50\inference.pdiparams
```

### 使用 OnnxRuntime 运行样例
```shell
.\Release\resnet50_test --model_file ..\resnet50\inference.pdmodel --params_file ..\resnet50\inference.pdiparams --use_ort=1
```

运行结束后，程序会将模型结果打印到屏幕，说明运行成功。
test
## 更多链接
- [Paddle Inference使用Quick Start！](https://paddle-inference.readthedocs.io/en/latest/introduction/quick_start.html)
- [Paddle Inference C++ Api使用](https://paddle-inference.readthedocs.io/en/latest/api_reference/cxx_api_index.html)
- [Paddle Inference Python Api使用](https://paddle-inference.readthedocs.io/en/latest/api_reference/python_api_index.html)