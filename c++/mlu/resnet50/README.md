# 寒武纪 MLU 运行 ResNet50 图像分类样例

ResNet50 样例展示了单输入模型在寒武纪 MLU 下的推理过程。运行步骤如下：

## 一、获取 Paddle Inference 预测库

当前仅支持通过源码编译的方式安装，源码编译方式参考 Paddle Inference 官网文档的硬件支持部分。

可以参考如下编译选项：

```bash
cd Paddle
mkdir build && cd build

# use the following cmake options
cmake .. -DPY_VERSION=3.10 \
		 -DPYTHON_EXECUTABLE=`which python3` \
		 -DWITH_GPU=OFF \
		 -DWITH_CUSTOM_DEVICE=ON \
		 -DON_INFER=ON \
		 -DWITH_TESTING=OFF \
		 -DCMAKE_CXX_FLAGS="-Wno-error -w"
make -j32
```

编译完成之后在编译目录下将会生成 Paddle Inference 的 C++ 预测库，即为编译目录下的 `paddle_inference_install_dir` 文件夹。文件夹目录如下示例：

```bash
# 检查编译目录下的 Python whl 包
Paddle/build/python/dist/
└── paddlepaddle_npu-0.0.0-cp37-cp37m-linux_x86_64.whl

# 检查编译目录下的 C++ 预测库，目录结构如下
Paddle/build/paddle_inference_install_dir
├── CMakeCache.txt
├── paddle
│   ├── include                                    # C++ 预测库头文件目录
│   │   ├── crypto
│   │   ├── experimental
│   │   ├── internal
│   │   ├── paddle_analysis_config.h
│   │   ├── paddle_api.h
│   │   ├── paddle_infer_contrib.h
│   │   ├── paddle_infer_declare.h
│   │   ├── paddle_inference_api.h                 # C++ 预测库头文件
│   │   ├── paddle_mkldnn_quantizer_config.h
│   │   ├── paddle_pass_builder.h
│   │   └── paddle_tensor.h
│   └── lib
│       ├── libpaddle_inference.a                  # C++ 静态预测库文件
│       └── libpaddle_inference.so                 # C++ 动态态预测库文件
├── third_party
│   ├── install                                    # 第三方链接库和头文件
│   │   ├── cryptopp
│   │   ├── gflags
│   │   ├── glog
│   │   ├── mkldnn
│   │   ├── mklml
│   │   ├── protobuf
│   │   ├── utf8proc
│   │   └── xxhash
│   └── threadpool
│       └── ThreadPool.h
└── version.txt                                    # 预测库版本信息

```

将编译好的`paddle_inference_install_dir`目录拷贝到`Paddle-Inference-Demo/c++/lib` 目录下。即可完成 Paddle Inference 预测库的安装。

## 二：准备 PaddleCustomDevice

PaddleCustomDevice 中包含了支持 MLU 的插件库，需要手动编译才能是能 MLU 设备。整体使用和编译方法可以参考 [PaddleCustomDevice-MLU](https://github.com/PaddlePaddle/PaddleCustomDevice/blob/develop/backends/mlu/README_cn.md)。若想适配 PaddleInference C++ API，则需修改编译选项：

```bash
# 给 CustomDevice 设置 PADDLE_INFERENCE_LIB_DIR 环境变量
export PADDLE_INFERENCE_LIB_DIR=/path/to/paddle_inference/paddle/lib

# 创建 customdevice 安装编译目录
cd PaddleCustomDevice/backends/mlu
mkdir build && cd build

# 配置并编译paddlecustomdevice 插件包
cmake .. -DWITH_PROFILE=OFF -DON_INFER=ON
make -j32
```

编译完成，需要设置 PaddleCustomDevice 的库目录到 CUSTOM_DEVICE_ROOT 环境变量，`export CUSTOM_DEVICE_ROOT=PaddleCustomDevice/backends/mlu/build`。该环境变量会在下面编译时做链接使用。

## 三：编译样例
 
- 文件`resnet50_test.cc` 为预测的样例程序（程序中的输入为固定值，如果您有opencv或其他方式进行数据读取的需求，需要对程序进行一定的修改）。    
- 脚本`compile.sh` 包含了第三方库、预编译库的信息配置。
- 脚本`run.sh` 为一键运行脚本。

编译前，需要根据自己的环境修改 `compile.sh` 中的相关代码配置依赖库：

```bash
# 根据预编译库中的version.txt信息判断是否将以下标记打开
WITH_MKL=ON  # 这里如果是 Aarch64 环境，则改为 OFF
WITH_ARM=OFF # 这里如果是 Aarch64 环境，则改为 ON
```

运行 `bash compile.sh` 编译样例。

## 四：运行样例

```shell
./build/resnet50_test --model_file resnet50/inference.pdmodel --params_file resnet50/inference.pdiparams
```
运行结束后，程序会将模型结果打印到屏幕，说明运行成功，预期得到如下的输出结果：

```bash
I0531 15:14:32.535790 23336 resnet50_test.cc:85] run avg time is 99605.8 ms
I0531 15:14:32.535897 23336 resnet50_test.cc:122] 0 : 2.67648e-43
I0531 15:14:32.535917 23336 resnet50_test.cc:122] 100 : 1.98485e-37
... ...
I0531 15:14:32.536034 23336 resnet50_test.cc:122] 800 : 3.80368e-25
I0531 15:14:32.536043 23336 resnet50_test.cc:122] 900 : 1.46269e-30
```

## 更多链接
- [Paddle Inference使用Quick Start！](https://paddle-inference.readthedocs.io/en/latest/introduction/quick_start.html)
- [Paddle Inference C++ Api使用](https://paddle-inference.readthedocs.io/en/latest/api_reference/cxx_api_index.html)
- [Paddle Inference Python Api使用](https://paddle-inference.readthedocs.io/en/latest/api_reference/python_api_index.html)
