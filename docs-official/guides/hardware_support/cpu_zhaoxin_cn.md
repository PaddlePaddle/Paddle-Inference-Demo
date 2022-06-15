# 兆芯 CPU 安装说明

Paddle Inference 支持基于兆芯 CPU 的推理部署, 当前仅支持通过源码编译的方式安装。

## 系统要求

当前 Paddle Inference 支持兆芯 CPU 在如下环境下的源码编译和安装部署：

| 处理器 | 操作系统 |
| ---- | ---- |
| 全系列CPU | Linux操作系统 (Ubuntu、CentOS) |

## 源码编译

**环境准备：** 请根据[编译依赖表](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/install/Tables.html)准备符合版本要求的依赖库，推荐使用飞桨官方镜像。

**第一步：** 从飞桨镜像库拉取编译镜像并启动容器，该镜像基于 Ubuntu 18.04 操作系统构建

```bash
# 拉取镜像
docker pull registry.baidubce.com/device/paddle-cpu:ubuntu18-x86_64-gcc82

# 启动容器
docker run -it --name paddle-dev -v `pwd`:/workspace \
     --network=host --shm-size=128G --workdir=/workspace \
     --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
     registry.baidubce.com/device/paddle-cpu:ubuntu18-x86_64-gcc82 /bin/bash
```

**第二步：** 下载 Paddle 源码并编译，CMAKE 编译选项含义请参见[编译选项表](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/install/Tables.html)

```bash
# 下载源码，默认 develop 分支
git clone https://github.com/PaddlePaddle/Paddle.git
cd Paddle

# 创建编译目录
mkdir build && cd build

# 执行cmake
cmake .. -DPY_VERSION=3 -DPYTHON_EXECUTABLE=`which python3` \
         -DWITH_TESTING=OFF -DON_INFER=ON -DWITH_XBYAK=OFF

# 使用以下命令来编译
make -j$(nproc)
```

**第三步：** 编译完成之后，请检查编译目录下的 Python whl 包 和 C++ 预测库是否正确生成

```bash
# 检查编译目录下的 Python whl 包
Paddle/build/python/dist/
└── paddlepaddle-0.0.0-cp37-cp37m-linux_x86_64.whl

# 检查编译目录下的 C++ 预测库，目录结构如下
Paddle/build/paddle_inference_install_dir
├── CMakeCache.txt
├── paddle
│   ├── include                                    # C++ 预测库头文件目录
│   │   ├── crypto
│   │   ├── experimental
│   │   ├── internal
│   │   ├── paddle_analysis_config.h
│   │   ├── paddle_api.h
│   │   ├── paddle_infer_contrib.h
│   │   ├── paddle_infer_declare.h
│   │   ├── paddle_inference_api.h                 # C++ 预测库头文件
│   │   ├── paddle_mkldnn_quantizer_config.h
│   │   ├── paddle_pass_builder.h
│   │   └── paddle_tensor.h
│   └── lib
│       ├── libpaddle_inference.a                  # C++ 静态预测库文件
│       └── libpaddle_inference.so                 # C++ 动态态预测库文件
├── third_party
│   ├── install                                    # 第三方链接库和头文件
│   │   ├── cryptopp
│   │   ├── gflags
│   │   ├── glog
│   │   ├── mkldnn
│   │   ├── mklml
│   │   ├── protobuf
│   │   ├── utf8proc
│   │   └── xxhash
│   └── threadpool
│       └── ThreadPool.h
└── version.txt                                    # 预测库版本信息
```

## 安装部署

### Python 安装部署

请参考以下步骤执行 Python 安装部署示例程序：

```bash
# 1) 安装源码编译生成的 Python whl 包
python3 -m pip install -U paddlepaddle-0.0.0-cp37-cp37m-linux_x86_64.whl

# 2) 进行简单功能的健康检查
python3 -c "import paddle; paddle.utils.run_check()"
# 预期得到如下输出结果
# Running verify PaddlePaddle program ...
# PaddlePaddle works well on 1 CPU.
# PaddlePaddle works well on 2 CPUs.
# PaddlePaddle is installed successfully! Let's start deep learning with PaddlePaddle now.

# 3) 下载 Paddle-Inference-Demo 示例代码，并进入 Python 代码目录
git clone https://github.com/PaddlePaddle/Paddle-Inference-Demo.git
cd Paddle-Inference-Demo/python/cpu/resnet50

# 4) 下载推理模型
wget https://paddle-inference-dist.bj.bcebos.com/Paddle-Inference-Demo/resnet50.tgz
tar xzf resnet50.tgz

# 5) 准备预测图片
wget https://paddle-inference-dist.bj.bcebos.com/inference_demo/python/resnet50/ILSVRC2012_val_00000247.jpeg

# 6) 运行 Python 预测程序
python3 infer_resnet.py --model_file=./resnet50/inference.pdmodel --params_file=./resnet50/inference.pdiparams
# 预期得到如下输出结果
# class index:  13
```

### C++ 安装部署

请参考以下步骤执行 C++ 安装部署示例程序：

```bash
# 1) 下载 Paddle-Inference-Demo 代码
git clone https://github.com/PaddlePaddle/Paddle-Inference-Demo.git

# 2) 拷贝源码编译生成的 C++ 预测库到 Paddle-Inference-Demo/c++/lib 目录下
cp -r Paddle/build/paddle_inference_install_dir Paddle-Inference-Demo/c++/lib/paddle_inference
# 拷贝完成之后 Paddle-Inference-Demo/c++/lib 目录结构如下
Paddle-Inference-Demo/c++/lib/
├── CMakeLists.txt
└── paddle_inference
    ├── CMakeCache.txt
    ├── paddle
    ├── third_party
    └── version.txt

# 3) 进入 C++ 示例代码目录，下载推理模型
cd Paddle-Inference-Demo/c++/cpu/resnet50/
wget https://paddle-inference-dist.bj.bcebos.com/Paddle-Inference-Demo/resnet50.tgz
tar xzf resnet50.tgz

# 4) 修改 compile.sh 编译文件，需根据 C++ 预测库的 version.txt 信息对以下的几处内容进行修改
WITH_GPU=OFF

# 5) 执行编译，编译完成之后在 build 下生成 resnet50_test 可执行文件
./compile.sh

# 6) 运行 C++ 预测程序
./build/resnet50_test --model_file resnet50/inference.pdmodel --params_file resnet50/inference.pdiparams
# 预期得到如下输出结果
# I0526 20:23:05.371094   452 resnet50_test.cc:79] run avg time is 807.533 ms
# I0526 20:23:05.371285   452 resnet50_test.cc:116] 0 : 0
# I0526 20:23:05.371311   452 resnet50_test.cc:116] 100 : 2.04168e-37
# ... ...
# I0526 20:23:05.371394   452 resnet50_test.cc:116] 800 : 3.85262e-25
# I0526 20:23:05.371405   452 resnet50_test.cc:116] 900 : 1.52393e-30
```

## 如何卸载

C++ 预测库无需卸载，Python whl 包请使用以下命令卸载：

```bash
python3 -m pip uninstall paddlepaddle
```
