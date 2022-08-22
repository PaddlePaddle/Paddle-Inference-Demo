# macOS 下从源码编译

## 环境准备

  macOS 版本 10.x/11.x/12.x (64 bit) (不支持 GPU 版本)

  Python 版本 3.6/3.7/3.8/3.9/3.10 (64 bit)

## 选择 CPU/GPU

目前仅支持在 macOS 环境下编译安装 CPU 版本的 Paddle Inference

## 安装步骤

在 macOS 的系统下有 2 种编译方式，推荐使用 Docker 编译。 Docker 环境中已预装好编译 Paddle 需要的各种依赖，相较本机编译环境更简单。

	1. Docker源码编译(目前仅支持 mac x86)
	2. 本机源码编译

### 使用 docker 编译（目前仅支持 x86）

[Docker](https://docs.docker.com/install/) 是一个开源的应用容器引擎。使用 Docker，既可以将 Paddle Inference 的安装&使用与系统环境隔离，也可以与主机共享 GPU、网络等资源

使用 Docker 编译时，您需要在本地主机上[安装 Docker](https://docs.docker.com/engine/install/)


请您按照以下步骤安装：

**1. 进入 Mac 的终端**

**2. 请首先选择您希望储存 PaddlePaddle 的路径，然后在该路径下使用以下命令将 PaddlePaddle 的源码从 github 克隆到本地当前目录下名为 Paddle 的文件夹中：**
```shell
git clone https://github.com/PaddlePaddle/Paddle.git
```

**3. 进入 Paddle 目录下：**
```shell
cd Paddle
```

**4. 拉取 PaddlePaddle 镜像**

对于国内用户，因为网络问题下载 docker 比较慢时，可使用百度提供的镜像：

CPU 版的 PaddlePaddle：
```shell
docker pull registry.baidubce.com/paddlepaddle/paddle:latest-dev
```

如果您的机器不在中国大陆地区，可以直接从 DockerHub 拉取镜像：

CPU 版的 PaddlePaddle：
```shell
docker pull paddlepaddle/paddle:latest-dev
```

**5. 创建并进入已配置好编译环境的 Docker 容器：**
```shell
docker run --name paddle-test -v $PWD:/paddle --network=host --privileged=true -it registry.baidubce.com/paddlepaddle/paddle:latest-dev /bin/bash
```
其中参数的意义为：

	--name paddle-test：为您创建的 Docker 容器命名为 paddle-test;
	-v $PWD:/paddle： 将当前目录挂载到 Docker 容器中的 /paddle 目录下（Linux 中 PWD 变量会展开为当前路径的绝对路径);
	--privileged=true: container 内的 root用户 拥有真正的 root 权限
	-it： 与宿主机保持交互状态;
	registry.baidubce.com/paddlepaddle/paddle:latest-dev：使用名为 registry.baidubce.com/paddlepaddle/paddle:latest-dev 的镜像创建 Docker 容器，/bin/bash 进入容器后启动 /bin/bash 命令。

注意： 请确保至少为 docker 分配 4g 以上的内存，否则编译过程可能因内存不足导致失败。

**6. 进入 Docker 后进入 paddle 目录下：**
```shell
cd /paddle
```

**7. 切换到较稳定版本下进行编译：**
```shell
git checkout [分支名]
```
例如：
```shell
git checkout release/2.3
```

**8. 创建并进入 /paddle/build 路径下：**
```shell
mkdir build && cd build
```

**9. 使用以下命令安装相关依赖：**

安装 protobuf 3.1.0。
```shell
pip3.7 install protobuf==3.1.0
```
注意：以上用 Python3.7 命令来举例，如您的 Python 版本为3.6/3.8/3.9/3.10，请将上述命令中的 pip3.7 改成对应的版本。

**10. 执行 cmake：**

对于需要编译 CPU 版本 PaddlePaddle 的用户（我们目前不支持 macOS 下 GPU 版本 PaddlePaddle 的编译）：
```shell
cmake .. -DPY_VERSION=3.7 -DWITH_TESTING=OFF -DWITH_MKL=ON -DWITH_GPU=OFF -DON_INFER=ON
```
请注意修改参数 -DPY_VERSION 为您希望编译使用的 python 版本, 例如 -DPY_VERSION=3.7 表示 python 版本为3.7

具体编译选项含义请参见编译选项表

| 选项 | 说明 | 默认值 |
|--|--|--|
| WITH_GPU | 是否支持GPU | ON |
| WITH_AVX | 是否编译含有AVX指令集的飞桨二进制文件 | ON |
| WITH_PYTHON | 是否内嵌PYTHON解释器并编译Wheel安装包 | ON |
| WITH_TESTING | 是否开启单元测试 | OFF |
| WITH_MKL | 是否使用MKL数学库，如果为否，将使用OpenBLAS | ON |
| WITH_SYSTEM_BLAS | 是否使用系统自带的BLAS | OFF |
| WITH_DISTRIBUTE | 是否编译带有分布式的版本 | OFF |
| WITH_BRPC_RDMA | 是否使用BRPC,RDMA作为RPC协议 | OFF |
| ON_INFER | 是否打开预测优化 | OFF |
| CUDA_ARCH_NAME | 是否只针对当前CUDA架构编译 | All:编译所有可支持的CUDA架构；Auto:自动识别当前环境的架构编译 |
| WITH_TENSORRT | 是否开启 TensorRT | OFF |
| TENSORRT_ROOT | TensorRT_lib的路径，该路径指定后会编译TRT子图功能eg:/paddle/nvidia/TensorRT/ | /usr |

**11. 执行编译**
使用多核编译
```shell
make -j4
```
注意： 编译过程中需要从 github 上下载依赖，请确保您的编译环境能正常从 github 下载代码。

**编译飞桨过程中可能会打开很多文件，如果编译过程中显示 “Too many open files” 错误时，请使用指令 ulimit -n 102400 来增大当前进程允许打开的文件数**
```shell
ulimit -n 102400
```

**11. 编译成功后可在 dist 目录找到生成的 .whl 包**
```shell
pip3 install python/dist/(安装包名字)
```

**12. 预测库编译**
```shell
make inference_lib_dist -j4
```
编译成功后，位于 build 目录下的 paddle_inference_install_dir 目录内生成 c++ 预测库和对应的头文件。


### 本机编译

请严格按照以下指令顺序执行

**1. 检查您的计算机和操作系统是否符合我们支持的编译标准：**
```shell
uname -m
```
并且在 `关于本机` 中查看系统版本

**2. 安装 Python 以及 pip：**

  建议不要使用 macOS 中自带 Python，使用 python [官方下载](https://www.python.org/downloads/mac-osx/) python3.6.x、python3.7.x、python3.8、python3.9、python3.10), pip以及其他的依赖，这将会使您高效编译


**3. (Only For Python3 )设置 Python 相关的环境变量:**

a. 首先使用
```shell
find `dirname $(dirname $(which python3.8))` -name "libpython3.*.dylib"
```
找到 Pythonlib 的路径（一般情况下弹出的第一个对应您需要使用的 python 的 dylib 路径），然后（下面 [python-lib-path] 替换为找到文件路径）

请注意，当您的 mac 上安装有多个 python 时请保证您正在使用的 python 是您希望使用的 python。

b. 设置 PYTHON_LIBRARIES：
```shell
export PYTHON_LIBRARY=[python-lib-path]
```

c. 其次使用找到 PythonInclude 的路径（通常是找到 [python-lib-path] 的上一级目录为同级目录的 include, 然后找到该目录下 python3.x 的路径），然后将下面 [python-include-path] 替换为找到路径。

d. 设置 PYTHON_INCLUDE_DIR:
```shell
export PYTHON_INCLUDE_DIRS=[python-include-path]
```

e. 设置系统环境变量路径：
```shell
export PATH=[python-bin-path]:$PATH
```
（这里 [python-bin-path] 为将 [python-lib-path] 的最后两级目录替换为 /bin/ 后的目录)

f. 设置动态库链接：
```shell
export LD_LIBRARY_PATH=[python-ld-path]
export DYLD_LIBRARY_PATH=[python-ld-path]
```
（这里 [python-ld-path] 为 [python-bin-path] 的上一级目录)

g. (可选）如果您是在 macOS 10.14 上编译 PaddlePaddle，请保证您已经安装了[对应版本](http://developer.apple.com/download)的 Xcode。


**4. 执行编译前请您确认您的环境中安装有编译依赖表中提到的相关[依赖](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/install/Tables.html#bianyiyilaibiao)，否则我们强烈推荐使用 [Homebrew](https://brew.sh/) 安装相关依赖。**

  macOS 下如果您未自行修改或安装过“编译依赖表”中提到的依赖，则仅需要使用 pip 安装 PyYAML, numpy，protobuf，wheel，使用 homebrew 安装 wget，swig, unrar，另外安装 cmake 即可

a. 这里特别说明一下 CMake 的安装：

CMake 我们支持 3.15 以上版本,推荐使用 CMake3.16, 请从 [CMake官方网站](https://cmake.org/files/v3.16/cmake-3.16.0-Darwin-x86_64.dmg)下载 CMake 镜像并安装


b. 如果您希望使用自己安装的 OpenBLAS 请 1）设置环境变量`OPENBLAS_ROOT`为您安装的 OpenBLAS 的路径；2）设置cmake编译选项`-DWITH_SYSTEM_BLAS=OFF`。

**5. 将 PaddlePaddle 的源码 clone 在当下目录下的 Paddle 的文件夹中，并进入 Padde 目录下：**
```shell
git clone https://github.com/PaddlePaddle/Paddle.git
cd Paddle
```

**6. 切换到较稳定 release 分支下进行编译：**
```shell
git checkout [分支名]
```
例如：
```shell
git checkout release/2.3
```

**7. 创建并进入 build 目录下：**
```shell
mkdir build && cd build
```

**8. 执行 cmake：**
具体编译选项含义请参见编译选项表

若您的机器为 Mac M1 机器，需要编译Arm架构、CPU版本PaddlePaddle：
```shell
cmake .. -DPY_VERSION=3.8 -DPYTHON_INCLUDE_DIR=${PYTHON_INCLUDE_DIRS} \
-DPYTHON_LIBRARY=${PYTHON_LIBRARY} -DWITH_GPU=OFF -DWITH_TESTING=OFF \
-DWITH_AVX=OFF -DWITH_ARM=ON -DCMAKE_BUILD_TYPE=Release -DWITH_INFER=ON
```
若编译arm架构的paddlepaddle，需要cmake版本为 3.19.2 以上

若您的机器不是Mac M1机器，需要编译x86_64架构、CPU版本PaddlePaddle：
```shell
cmake .. -DPY_VERSION=3.8 -DPYTHON_INCLUDE_DIR=${PYTHON_INCLUDE_DIRS} \
-DPYTHON_LIBRARY=${PYTHON_LIBRARY} -DWITH_GPU=OFF -DWITH_TESTING=OFF  -DCMAKE_BUILD_TYPE=Release -DWITH_INFER=ON
```

**9. 使用以下命令来编译：**

若您的机器为 Mac M1 机器，需要编译 Arm 架构、CPU 版本 PaddlePaddle：
```shell
make TARGET=ARMV8 -j4
```

若您的机器不是 Mac M1 机器，需要编译 x86_64 架构、CPU版本 PaddlePaddle：
```shell
make -j4
```

**10. 编译成功后可在 dist 目录找到生成的 .whl 包**
```shell
pip3 install python/dist/[wheel 安装包]
```

**11. 在 paddle_inference_install_dir 目录下有生成的 paddle inference c++ 库和头文件等**

**12. 验证安装**
安装完成后你可以使用 python 进入 python 解释器，输入：
```shell
import paddle
paddle.utils.run_check()
```
如果出现 `PaddlePaddle is installed successfully!`，说明你已成功安装。

恭喜，至此你已完成 Paddle Inference 的编译安装