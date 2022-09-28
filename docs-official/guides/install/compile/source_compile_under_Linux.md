# Linux 下从源码编译

## 环境准备

Linux 版本 (64 bit)

    CentOS 6 (不推荐，不提供编译出现问题时的官方支持)
    CentOS 7
    Ubuntu 14.04 (不推荐，不提供编译出现问题时的官方支持)
    Ubuntu 16.04
    Ubuntu 18.04

Python 版本 3.6/3.7/3.8/3.9/3.10 (64 bit)

## 选择 CPU/GPU

如果您的计算机没有 NVIDIA GPU，请安装 CPU 版本的 PaddlePaddle。
如果您的计算机有 NVIDIA GPU，请确保满足以下条件以编译 GPU 版 PaddlePaddle：

    CUDA 工具包 10.2 配合 cuDNN 7 (cuDNN 版本>=7.6.5)
    CUDA 工具包 11.0 配合 cuDNN v8.0.4
    CUDA 工具包 11.1 配合 cuDNN v8.1.1
    CUDA 工具包 11.2 配合 cuDNN v8.1.1
    GPU 运算能力超过 3.5 的硬件设备

您可参考 NVIDIA 官方文档了解 CUDA 和 cuDNN 的安装流程和配置方法，请见 [CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)，[cuDNN](https://docs.nvidia.com/deeplearning/cuDNN/install-guide/)

## 安装步骤

在 Linux 的系统下有 2 种编译方式，推荐使用 Docker 编译。 Docker 环境中已预装好编译 Paddle 需要的各种依赖，相较本机编译环境更简单。

    1. 使用 Docker 编译（不提供在 CentOS 6 下编译中遇到问题的支持）
    2. 本机编译（不提供在 CentOS 6 下编译中遇到问题的支持）

### 使用 docker 编译

Docker 是一个开源的应用容器引擎。使用 Docker，既可以将 PaddlePaddle 的安装&使用与系统环境隔离，也可以与主机共享 GPU、网络等资源

使用 Docker 编译时，您需要：

    1. 在本地主机上安装 Docker
    2. 如需在 Linux 开启 GPU 支持，请安装 nvidia-docker

请您按照以下步骤编译安装：

**1. 请首先选择您希望储存 PaddlePaddle 的路径，然后在该路径下使用以下命令将 PaddlePaddle 的源码从 github 克隆到本地当前目录下名为 Paddle 的文件夹中**
```shell
git clone https://github.com/PaddlePaddle/Paddle.git
```

**2. 进入 Paddle 目录下**
```shell
cd Paddle
```

**3. 拉取 PaddlePaddle 镜像**

对于国内用户，因为网络问题下载 docker 比较慢时，可使用百度提供的镜像：

CPU 版的 PaddlePaddle：
```shell
docker pull registry.baidubce.com/paddlepaddle/paddle:latest-dev
```

GPU 版的 PaddlePaddle：
```shell
nvidia-docker pull registry.baidubce.com/paddlepaddle/paddle:latest-dev-cuda11.2-cuDNN8-gcc82
```

如果您的机器不在中国大陆地区，可以直接从 DockerHub 拉取镜像：

CPU 版的 PaddlePaddle：
```shell
docker pull paddlepaddle/paddle:latest-dev
```

GPU 版的 PaddlePaddle：
```shell
nvidia-docker pull paddlepaddle/paddle:latest-dev-cuda11.2-cuDNN8-gcc82
```

上例中，latest-dev-cuda11.2-cuDNN8-gcc82 仅作示意用，表示安装 GPU 版的镜像。如果您还想安装其他 cuda/cuDNN 版本的镜像，可以将其替换成 latest-gpu-cuda10.1-cuDNN7-gcc82-dev、latest-gpu-cuda10.1-cuDNN7-gcc54-dev 等。 您可以访问 [DockerHub](https://hub.docker.com/r/paddlepaddle/paddle/tags/) 获取与您机器适配的镜像。

**4. 创建并进入已配置好编译环境的 Docker 容器**

编译 CPU 版本的 PaddlePaddle：

    docker run --name paddle-test -v $PWD:/paddle --network=host --privileged=true -it registry.baidubce.com/paddlepaddle/paddle:latest-dev /bin/bash

其中参数的意义为：

    --name paddle-test：为您创建的 Docker 容器命名为 paddle-test;
    -v $PWD:/paddle： 将当前目录挂载到 Docker 容器中的 /paddle 目录下（Linux 中 PWD 变量会展开为当前路径的绝对路径);
    --privileged=true: container 内的 root 用户 拥有真正的 root 权限
    -it： 与宿主机保持交互状态;
    registry.baidubce.com/paddlepaddle/paddle:latest-dev：使用名为 registry.baidubce.com/paddlepaddle/paddle:latest-dev 的镜像创建 Docker 容器，/bin/bash 进入容器后启动 /bin/bash 命令。

编译 GPU 版本的 PaddlePaddle：

    nvidia-docke run --name paddle-test -v $PWD:/paddle --network=host --privileged=true -it registry.baidubce.com/paddlepaddle/paddle:latest-dev /bin/bash

注意： 请确保至少为 docker 分配 4g 以上的内存，否则编译过程可能因内存不足导致失败。

**5. 进入 Docker 后进入 paddle 目录下**
```shell
cd /paddle
```

**6. 切换到较稳定版本下进行编译**
```shell
git checkout [分支名]
```
例如：
```shell
git checkout release/2.3
```

**7. 创建并进入 /paddle/build 路径下**
```shell
mkdir -p /paddle/build && cd /paddle/build
```

**8. 使用以下命令安装相关依赖**

安装 protobuf。
```shell
pip3.7 install protobuf
```
注意：以上用 Python3.7 命令来举例，请将上述命令中的 pip3.7 改成对应的版本。

**9. 执行 cmake**

编译 CPU 版本：
```shell
cmake .. -DPY_VERSION=3.7 -DWITH_TESTING=OFF -DWITH_MKL=ON -DWITH_GPU=OFF -DON_INFER=ON
```

编译 GPU 版本：
```shell
cmake .. -DPY_VERSION=3.7 -DWITH_TESTING=OFF -DWITH_MKL=ON -DWITH_GPU=ON -DON_INFER=ON
```

使用 TensorRT：

如果想使用 TensorRT 进行推理，首先需要根据自己的需求下载对应版本的 [TensorRT GA build](https://developer.nvidia.com/nvidia-tensorrt-download),
下载解压后，在 cmake 中开启 WITH_TENSORRT， 并通过 TENSORRT_ROOT 指定刚刚解压的 TensorRT_lib 的路径。假设下载的 TensorRT lib 解压
目录为 /paddle/nvidia/TensorRT/， cmake 编译指令如下：
```shell
cmake .. -DPY_VERSION=3.7 -DWITH_TESTING=OFF -DWITH_MKL=ON -DWITH_GPU=ON -DON_INFER=ON \
                    -DWITH_TENSORRT=ON -DTENSORRT_ROOT=/paddle/nvidia/TensorRT/
```

更多 cmake 参数可以查看 cmake 参数表：

| 选项 | 说明 | 默认值 |
|--|--|--|
| WITH_GPU | 是否支持 GPU | ON |
| WITH_AVX | 是否编译含有 AVX 指令集的飞桨二进制文件 | ON |
| WITH_PYTHON | 是否内嵌 PYTHON 解释器并编译 Wheel 安装包 | ON |
| WITH_TESTING | 是否开启单元测试 | OFF |
| WITH_MKL | 是否使用 MKL 数学库，如果为否，将使用 OpenBLAS | ON |
| WITH_SYSTEM_BLAS | 是否使用系统自带的 BLAS | OFF |
| WITH_DISTRIBUTE | 是否编译带有分布式的版本 | OFF |
| WITH_BRPC_RDMA | 是否使用 BRPC,RDMA 作为 RPC 协议 | OFF |
| ON_INFER | 是否打开推理优化 | OFF |
| CUDA_ARCH_NAME | 是否只针对当前 CUDA 架构编译 | All:编译所有可支持的 CUDA 架构；Auto:自动识别当前环境的架构编译 |
| WITH_TENSORRT | 是否开启 TensorRT | OFF |
| TENSORRT_ROOT | TensorRT_lib 的路径，该路径指定后会编译 TensorRT 子图功能 eg:/paddle/nvidia/TensorRT/ | /usr |

**10. 执行编译**
```shell
make -j4
```

编译飞桨过程中可能会打开很多文件，如果编译过程中显示 “Too many open files” 错误时，请使用指令 ulimit -n 102400 来增大当前进程允许打开的文件数**
```shell
ulimit -n 102400
```
注意： 编译过程中需要从 github 上下载依赖，请确保您的编译环境能正常从 github 下载代码。

**11. 编译成功后可在 dist 目录找到生成的 .whl 包**
```shell
pip3 install python/dist/[wheel 包名字]
```

**12. 推理库编译**
```shell
make inference_lib_dist -j4
```
编译成功后，所有产出均位于 build 目录下的 paddle_inference_install_dir 目录内。



### 本机编译
本机编译与 docker 编译的区别只有环境准备不同， docker 中已经配置好了相关环境，本机编译中，需要用户自己配置[编译依赖项](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/install/Tables.html#bianyiyilaibiao)。


**1. 安装必要的工具**

以 Ubuntu 上为例, 安装编译依赖项可通过如下命令：

```shell
sudo apt-get install gcc g++ make cmake git vim unrar python3 python3-dev python3-pip swig wget  libopencv-dev
pip3 install numpy protobuf wheel setuptools
```

若需启用 CUDA 加速，需准备 CUDA、cuDNN。请参考 NVIDIA 官网文档了解 CUDA 和 cuDNN 的安装流程和配置方法，请见 [CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)，[cuDNN](https://docs.nvidia.com/deeplearning/cuDNN/install-guide/), 版本对应关系如下表所示:

|CUDA 版本|cuDNN 版本| TensorRT 版本|
|---|---|---|
|10.2|7.6|7|
|11.0|8.0|7|
|11.2|8.2|8|

以 CUDA 11.3，cuDNN 8.2 为例配置 CUDA 环境。
```shell
# cuda
sh cuda_11.3.0_465.19.01_linux.run
export PATH=/usr/local/cuda-11.3/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11.3/${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# cuDNN
tar -xzvf cudnn-11.3-linux-x64-v8.2.1.32.tgz
sudo cp -a cuda/include/cuDNN.h /usr/local/cuda/include/
sudo cp -a cuda/lib64/libcuDNN* /usr/local/cuda/lib64/
```

**2. 编译**

2.1) 下载 Paddle

使用 Git 将飞桨代码克隆到本地，并进入目录，切换到稳定版本（git tag 显示的标签名，如 release/2.3）。 飞桨使用 develop 分支进行最新特性的开发，使用 release 分支发布稳定版本。在 GitHub 的 Releases 选项卡中，可以看到飞桨版本的发布记录。

```shell
git clone https://github.com/PaddlePaddle/Paddle.git
cd Paddle
git checkout release/2.3
```

2.2）cmake

下面以 GPU 版本为例说明编译命令。其他环境可以参考“ CMake 编译选项表”修改对应的 cmake 选项。比如，若编译 CPU 版本，请将 WITH_GPU 设置为 OFF。
```shell
# 创建并进入 build 目录
mkdir build_cuda && cd build_cuda
# 执行 cmake 指令
cmake .. -DPY_VERSION=3 \
        -DWITH_TESTING=OFF \
        -DWITH_MKL=ON \
        -DWITH_GPU=ON \
        -DON_INFER=ON \
        ..
```

2.3） 使用 make 编译

```shell
make -j4
```
更多 cmake 参数可以查看 cmake 参数表：

| 选项 | 说明 | 默认值 |
|--|--|--|
| WITH_GPU | 是否支持 GPU | ON |
| WITH_AVX | 是否编译含有 AVX 指令集的飞桨二进制文件 | ON |
| WITH_PYTHON | 是否内嵌 PYTHON 解释器并编译 Wheel 安装包 | ON |
| WITH_TESTING | 是否开启单元测试 | OFF |
| WITH_MKL | 是否使用 MKL 数学库，如果为否，将使用 OpenBLAS | ON |
| WITH_SYSTEM_BLAS | 是否使用系统自带的 BLAS | OFF |
| WITH_DISTRIBUTE | 是否编译带有分布式的版本 | OFF |
| WITH_BRPC_RDMA | 是否使用 BRPC,RDMA 作为 RPC 协议 | OFF |
| ON_INFER | 是否打开推理优化 | OFF |
| CUDA_ARCH_NAME | 是否只针对当前 CUDA 架构编译 | All:编译所有可支持的 CUDA 架构；Auto:自动识别当前环境的架构编译 |
| WITH_TENSORRT | 是否开启 TensorRT | OFF |
| TENSORRT_ROOT | TensorRT_lib 的路径，该路径指定后会编译 TensorRT 子图功能 eg:/paddle/nvidia/TensorRT/ | /usr |

**3. 安装 Wheel 包**

编译成功后可在 dist 目录找到生成的 .whl 包
```shell
pip3 install python/dist/[wheel 包名字]
```

**4. 编译 C++推理库（按需）**
```shell
make inference_lib_dist -j4
```
编译成功后，所有产出均位于 build 目录下的 paddle_inference_install_dir 目录内。

编译飞桨过程中可能会打开很多文件，如果编译过程中显示 “Too many open files” 错误时，请使用指令 ulimit -n 102400 来增大当前进程允许打开的文件数
```shell
ulimit -n 102400
```

**5. 验证安装**

安装完成后你可以使用 python 进入 python 解释器，输入：
```shell
import paddle
paddle.utils.run_check()
```
如果出现 `PaddlePaddle is installed successfully!`，说明你已成功安装。

恭喜，至此你已完成 Paddle Inference 的编译安装
