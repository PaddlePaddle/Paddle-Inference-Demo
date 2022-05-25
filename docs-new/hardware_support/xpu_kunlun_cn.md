# 昆仑 XPU 安装说明

Paddle Inference 支持基于昆仑 XPU 的推理部署, 当前仅支持通过源码编译的方式安装。

## 系统要求

当前 Paddle Inference 支持昆仑 XPU 在如下环境下的源码编译和安装部署：

| 芯片型号 | 操作系统 |
| ---- | ---- |
| 昆仑1代芯片（K100、K200) | Linux操作系统 (Ubuntu、CentOS), 麒麟 V10 |
| 昆仑2代芯片 (R200、R300) | Linux操作系统 (Ubuntu、CentOS), 麒麟 V10 |

## 源码编译

**环境准备：** 请根据[编译依赖表](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/install/Tables.html)准备符合版本要求的依赖库，推荐使用飞桨官方镜像，否则请参考操作系统使用文档如[麒麟技术文档](https://eco.kylinos.cn/document/science.html)。

### X86_64 编译安装

**第一步：** 从飞桨镜像库拉取编译镜像并启动容器，该镜像基于 Ubuntu 18.04 操作系统构建

```bash
# 拉取镜像
docker pull registry.baidubce.com/device/paddle-dev:xpu-x86_64

# 启动容器，注意这里需要添加参数 --privileged，否则无法在容器内查看设备
docker run -it --name paddle-dev -v `pwd`:/workspace \
           --shm-size=128G --network=host --privileged \
           --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
           registry.baidubce.com/device/paddle-dev:xpu-x86_64 /bin/bash

# 容器内检查设备情况
xpu_smi
# 预期获得如下输出结果
Runtime Version: 4.0
Driver Version: 4.0
  DEVICES
-------------------------------------------------------------------------------------------------------
| DevID |   PCI Addr   | Model |        SN        |    INODE   | UseRate |     L3     |    Memory     |
-------------------------------------------------------------------------------------------------------
|     0 | 0000:06:00.0 | K200  | 0200210302000998 | /dev/xpu0  |     0 % | 14 / 16 MB | 504 / 8064 MB |
|     1 | 0000:06:00.0 | K200  | 0200210302000998 | /dev/xpu1  |     0 % | 14 / 16 MB | 504 / 8064 MB |
|     2 | 0001:03:00.0 | K200  | 0200210202001104 | /dev/xpu2  |     0 % |  0 / 16 MB |   0 / 8064 MB |
|     3 | 0001:03:00.0 | K200  | 0200210202001104 | /dev/xpu3  |     0 % |  0 / 16 MB |   0 / 8064 MB |
-------------------------------------------------------------------------------------------------------
  PROCESSES
-------------------------------------------------
| DevID | PID | Streams | L3 | Memory | Command |
-------------------------------------------------
-------------------------------------------------
```

**第二步：** 下载 Paddle 源码并编译，CMAKE 编译选项含义请参见[编译选项表](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/install/Tables.html)

```bash
# 下载源码，默认 develop 分支
git clone https://github.com/PaddlePaddle/Paddle.git
cd Paddle

# 创建编译目录
mkdir build && cd build

# 执行cmake
cmake .. -DPY_VERSION=3 -DPYTHON_EXECUTABLE=`which python3` -DWITH_XPU=ON \
         -DON_INFER=ON -DWITH_TESTING=OFF -DWITH_XBYAK=OFF

# 使用以下命令来编译
make -j$(nproc)
```

### Aarch64 编译安装

**第一步：** 从飞桨镜像库拉取编译镜像并启动容器，该镜像基于麒麟 V10 操作系统构建

```bash
# 拉取镜像
docker pull registry.baidubce.com/device/paddle-dev:xpu-aarch64

# 启动容器，注意这里需要添加参数 --privileged，否则无法在容器内查看设备
docker run -it --name paddle-dev -v `pwd`:/workspace \
           --shm-size=128G --network=host --privileged \
           --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
           registry.baidubce.com/device/paddle-dev:xpu-aarch64 /bin/bash

# 容器内检查设备情况
xpu_smi
# 预期获得如下输出结果
Runtime Version: 4.0
Driver Version: 4.0
  DEVICES
---------------------------------------------------------------------------------------------------------
| DevID |   PCI Addr   | Model |        SN        |    INODE   | UseRate |     L3     |     Memory      |
---------------------------------------------------------------------------------------------------------
|     0 | 0000:03:00.0 | R200  | 02K00Y6219V00013 | /dev/xpu0  |    12 % | 63 / 63 MB | 4146 / 13568 MB |
|     1 | 0001:03:00.0 | R200  | 02K00Y621AV0001Y | /dev/xpu1  |     0 % |  0 / 63 MB |    0 / 13568 MB |
---------------------------------------------------------------------------------------------------------
  PROCESSES
-------------------------------------------------
| DevID | PID | Streams | L3 | Memory | Command |
-------------------------------------------------
-------------------------------------------------
```

**第二步：** 下载 Paddle 源码并编译，CMAKE 编译选项含义请参见[编译选项表](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/install/Tables.html)

```bash
# 下载源码，默认 develop 分支
git clone https://github.com/PaddlePaddle/Paddle.git
cd Paddle

# 创建编译目录
mkdir build && cd build

# 执行cmake
cmake .. -DPY_VERSION=3 -DPYTHON_EXECUTABLE=`which python3` -DWITH_XPU=ON \
         -DON_INFER=ON -DWITH_TESTING=OFF -DWITH_XBYAK=OFF \
         -DWITH_ARM=ON -DWITH_AARCH64=ON

# 使用以下命令来编译
make TARGET=ARMV8 -j$(nproc)
```

### 编译后检查

编译完成之后，请检查编译目录下的 Python whl 包 和 C++ 预测库是否正确生成。以 Aarch64 环境为例，生成的的目录结构如下所示：

```bash
# 检查编译目录下的 Python whl 包
Paddle/build/python/dist/
└── paddlepaddle_xpu-0.0.0-cp37-cp37m-linux_aarch64.whl

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
│   ├── install
│   │   ├── cryptopp
│   │   ├── gflags
│   │   ├── glog
│   │   ├── openblas
│   │   ├── protobuf
│   │   ├── utf8proc
│   │   ├── xpu
│   │   └── xxhash
│   └── threadpool
│       └── ThreadPool.h
└── version.txt                                    # 预测库版本信息
```

## 安装部署

本章节以 Aarch64 环境为例说明 Paddle Inference Demo 的安装部署示例：

### Python 安装部署

请参考以下步骤执行 Python 安装部署示例程序：

```bash
# 1) 安装源码编译生成的 Python whl 包
python3 -m pip install -U paddlepaddle_xpu-0.0.0-cp37-cp37m-linux_aarch64.whl

# 2) 进行简单功能的健康检查
python3 -c "import paddle; paddle.utils.run_check()"
# 预期得到如下输出结果
# Running verify PaddlePaddle program ...
# PaddlePaddle works well on 1 XPU.
# PaddlePaddle works well on 4 XPUs.
# PaddlePaddle is installed successfully! Let's start deep learning with PaddlePaddle now.

# 3) 下载 Paddle-Inference-Demo 示例代码，并进入 Python 代码目录
git clone https://github.com/PaddlePaddle/Paddle-Inference-Demo.git
cd Paddle-Inference-Demo/python/resnet50

# 4) 下载推理模型
wget https://paddle-inference-dist.bj.bcebos.com/Paddle-Inference-Demo/resnet50.tgz
tar xzf resnet50.tgz

# 5) 准备预测图片
wget https://paddle-inference-dist.bj.bcebos.com/inference_demo/python/resnet50/ILSVRC2012_val_00000247.jpeg

# 6) 运行 Python 预测程序，注意这里需要设置 --use_xpu=1
python3 infer_resnet.py --model_file=./resnet50/inference.pdmodel \
                        --params_file=./resnet50/inference.pdiparams --use_xpu=1
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
cd Paddle-Inference-Demo/c++/resnet50/
wget https://paddle-inference-dist.bj.bcebos.com/Paddle-Inference-Demo/resnet50.tgz
tar xzf resnet50.tgz

# 4) 修改 compile.sh 编译文件，需根据 C++ 预测库的 version.txt 信息对以下的几处内容进行修改
WITH_MKL=OFF # 这里如果是 X86_64 环境，则改为 ON
WITH_GPU=OFF
WITH_ARM=ON # 这里如果是 X86_64 环境，则改为 OFF
WITH_XPU=ON

# 5) 执行编译，编译完成之后在 build 下生成 resnet50_test 可执行文件
./compile.sh

# 6) 运行 C++ 预测程序，注意这里需要设置 --use_xpu
./build/resnet50_test --model_file resnet50/inference.pdmodel \
                      --params_file resnet50/inference.pdiparams --use_xpu
# 预期得到如下输出结果
# W0525 20:56:43.035851 95178 xpu_context.cc:89] Please NOTE: xpu device: 0
# W0525 20:56:43.035950 95178 device_context.cc:310] Please NOTE: xpu device: 0
# I0525 20:56:43.083045 95178 resnet50_test.cc:79] run avg time is 46.773 ms
# I0525 20:56:43.083097 95178 resnet50_test.cc:116] 0 : 6.93194e-15
# I0525 20:56:43.083169 95178 resnet50_test.cc:116] 100 : 6.93194e-15
# ... ...
# I0525 20:56:43.083432 95178 resnet50_test.cc:116] 800 : 6.93194e-15
# I0525 20:56:43.083436 95178 resnet50_test.cc:116] 900 : 6.93194e-15
```

## 如何卸载

C++ 预测库无需卸载，Python whl 包请使用以下命令卸载：

```bash
python3 -m pip uninstall paddlepaddle-xpu
```
