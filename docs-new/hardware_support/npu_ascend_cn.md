# 昇腾 NPU 安装说明

Paddle Inference 支持基于 华为昇腾 NPU 的推理部署, 当前仅支持通过源码编译的方式安装。

## 系统要求

当前 Paddle Inference 支持 华为昇腾 NPU 在如下环境下的源码编译和安装部署：

| 芯片型号 | 操作系统 | SDK 版本 |
| ---- | ---- | ---- |
| Ascend 910  | Ubuntu 18.04 | CANN 5.0.4.alpha005 |

## 源码编译

**环境准备：** 请根据[编译依赖表](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/install/Tables.html)准备符合版本要求的依赖库，推荐使用飞桨官方镜像，或者根据 [CANN 文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/deploy504alpha5) 来准备相应的运行环境。

**第一步：** 从飞桨镜像库拉取编译镜像，启动容器并在容器内检查设备情况

```bash
# 拉取镜像
docker pull registry.baidubce.com/device/paddle-npu:cann504-x86_64-gcc75

# 启动容器，注意这里的参数 --device，容器仅映射设备ID为2到3的2张NPU卡，如需映射其他卡相应增改设备ID号即可
docker run -it --name paddle-dev -v `pwd`:/workspace  \
       --workdir=/workspace --pids-limit 409600 \
       --privileged --network=host --shm-size=128G \
       -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
       -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
       -v /usr/local/dcmi:/usr/local/dcmi \
       registry.baidubce.com/device/paddle-npu:cann504-x86_64-gcc75 /bin/bash

# 容器内检查设备情况
npu-smi info
# 预期获得如下输出结果
+-------------------------------------------------------------------------------------------+
| npu-smi 21.0.4                   Version: 21.0.4                                          |
+----------------------+---------------+----------------------------------------------------+
| NPU   Name           | Health        | Power(W)    Temp(C)           Hugepages-Usage(page)|
| Chip                 | Bus-Id        | AICore(%)   Memory-Usage(MB)  HBM-Usage(MB)        |
+======================+===============+====================================================+
| 0     910A           | OK            | 70.9        42                15   / 15            |
| 0                    | 0000:C1:00.0  | 0           839  / 15170      1    / 32768         |
+======================+===============+====================================================+
| 1     910A           | OK            | 67.2        36                15   / 15            |
| 0                    | 0000:81:00.0  | 0           1274 / 15171      1    / 32768         |
+======================+===============+====================================================+
```

**第二步**：下载Paddle源码并编译，CMAKE编译选项含义请参见[编译选项表](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/install/Tables.html#Compile)

```bash
# 下载源码，默认 develop 分支
git clone https://github.com/PaddlePaddle/Paddle.git
cd Paddle

# 创建编译目录
mkdir build && cd build

# 执行cmake
cmake .. -DPY_VERSION=3.7 -DPYTHON_EXECUTABLE=`which python3` -DON_INFER=ON \
         -DWITH_ASCEND=OFF -DWITH_ASCEND_CL=ON -DWITH_ASCEND_INT64=ON  \
         -DWITH_ASCEND_CXX11=ON -DWITH_TESTING=OFF \
         -DCMAKE_CXX_FLAGS="-Wno-error -w"

# 使用以下命令来编译
make -j$(nproc)
```

**第三步：** 编译完成之后，请检查编译目录下的 Python whl 包 和 C++ 预测库是否正确生成

```bash
# 检查编译目录下的 Python whl 包
Paddle/build/python/dist/
└── paddlepaddle_npu-0.0.0-cp37-cp37m-linux_x86_64.whl

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
python3 -m pip install -U paddlepaddle_npu-0.0.0-cp37-cp37m-linux_x86_64.whl

# 2) 进行简单功能的健康检查
python3 -c "import paddle; paddle.utils.run_check()"
# 预期得到如下输出结果
# Running verify PaddlePaddle program ...
# PaddlePaddle works well on 1 NPU.
# PaddlePaddle works well on 4 NPUs.
# PaddlePaddle is installed successfully! Let's start deep learning with PaddlePaddle now.

# 3) 下载 Paddle-Inference-Demo 示例代码，并进入 Python 代码目录
git clone https://github.com/PaddlePaddle/Paddle-Inference-Demo.git
cd Paddle-Inference-Demo/python/resnet50

# 4) 下载推理模型
wget https://paddle-inference-dist.bj.bcebos.com/Paddle-Inference-Demo/resnet50.tgz
tar xzf resnet50.tgz

# 5) 准备预测图片
wget https://paddle-inference-dist.bj.bcebos.com/inference_demo/python/resnet50/ILSVRC2012_val_00000247.jpeg

# 6) 运行 Python 预测程序，注意这里需要设置 --use_gpu=1
python3 infer_resnet.py --model_file=./resnet50/inference.pdmodel \
                       --params_file=./resnet50/inference.pdiparams --use_npu=1
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

# 4) 修改 compile.sh 编译文件，需根据 C++ 预测库的 version.txt 信息对以下内容进行修改
WITH_GPU=OFF
WITH_NPU=ON
ASCEND_LIB=/usr/local/Ascend # 这里请根据实际 CANN 安装路径修改

# 5) 执行编译，编译完成之后在 build 下生成 resnet50_test 可执行文件
./compile.sh

# 6) 运行 C++ 预测程序，注意这里需要设置 --use_gpu
./build/resnet50_test --model_file resnet50/inference.pdmodel \
                      --params_file resnet50/inference.pdiparams --use_npu
# 预期得到如下输出结果
# I0531 15:14:32.535790 23336 resnet50_test.cc:85] run avg time is 99605.8 ms
# I0531 15:14:32.535897 23336 resnet50_test.cc:122] 0 : 2.67648e-43
# I0531 15:14:32.535917 23336 resnet50_test.cc:122] 100 : 1.98485e-37
# ... ...
# I0531 15:14:32.536034 23336 resnet50_test.cc:122] 800 : 3.80368e-25
# I0531 15:14:32.536043 23336 resnet50_test.cc:122] 900 : 1.46269e-30
```

## 如何卸载

C++ 预测库无需卸载，Python whl 包请使用以下命令卸载：

```bash
python3 -m pip uninstall paddlepaddle-npu
```
