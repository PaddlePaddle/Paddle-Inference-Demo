# Windows 下从源码编译 

## 环境准备

	Windows 7/8/10 专业版/企业版 (64 bit)
	Python 版本 3.6/3.7/3.8/3.9/3.10 (64 bit)
	Visual Studio 2017 社区版/专业版/企业版

## 选择CPU/GPU

	如果你的计算机硬件没有 NVIDIA GPU，请编译 CPU 版本的 Paddle Inference 预测库
	如果你的计算机硬件有 NVIDIA GPU，推荐编译 GPU 版本的 Paddle Inference 预测库，建议安装 CUDA 10.2/11.0/11.1/11.2

## 本机编译过程

**1. 安装必要的工具 cmake, git, python, Visual studio 2017：**

- cmake：建议安装 CMake3.17 版本, 官网下载[链接](https://cmake.org/files/v3.17/cmake-3.17.0-win64-x64.msi)。安装时注意勾选 `Add CMake to the system PATH for all users`，将 CMake 添加到环境变量中。

- git：官网下载[链接](https://github.com/git-for-windows/git/releases/download/v2.35.1.windows.2/Git-2.35.1.2-64-bit.exe)，使用默认选项安装。
	
- python：官网[链接](https://www.python.org/downloads/windows/)，可选择 3.6/3.7/3.8/3.9/3.10 中任一版本的 Windows installer(64-bit) 安装。安装时注意勾选 `Add Python 3.x to PATH`，将 Python 添加到环境变量中。

- Visual studio 2017：官网[链接](https://visualstudio.microsoft.com/zh-hans/vs/older-downloads/#visual-studio-2017-and-other-products)，需要登录后下载，建议下载 Community 社区版。在安装时需要在工作负荷一栏中勾选 使用 `C++ 的桌面开发` 和 `通用 Windows 平台开发`，并在语言包一栏中选择 `英语`。

**2. 在 Windows 桌面下方的搜索栏中搜索 `x64 Native Tools Command Prompt for VS 2017` 或 `适用于 VS 2017 的 x64 本机工具命令提示符`，右键以管理员身份打开终端。之后的命令均在该终端中执行。**

**3.使用 pip 命令安装 Python 依赖：**

通过 `python --version` 检查默认 python 版本是否是预期版本，因为你的计算机可能安装有多个 python，你可通过修改系统环境变量的顺序来修改默认 Python 版本。

安装 numpy, protobuf, wheel, ninja
```shell
pip3 install numpy protobuf wheel ninja
```

**4. 创建编译 Paddle 的文件夹（例如 D:\workspace），进入该目录并下载源码：**
```
mkdir D:\workspace && cd /d D:\workspace
git clone https://github.com/PaddlePaddle/Paddle.git
cd Paddle
```
**5. 切换到 2.3 分支下进行编译：**
```shell
git checkout release/2.3
```
**6. 创建名为 build 的目录并进入：**
```shell
mkdir build
cd build
```

**7. 执行 cmake：**

编译 CPU 版本的 Paddle Inference：
```shell
cmake .. -GNinja -DWITH_GPU=OFF  -DCMAKE_BUILD_TYPE=Release -DWITH_UNITY_BUILD=ON -DWITH_TESTING=OFF -DON_INFER=ON
```

编译 GPU 版本的 Paddle Inference：
```shell
cmake .. -GNinja -DWITH_GPU=ON  -DCMAKE_BUILD_TYPE=Release -DWITH_UNITY_BUILD=ON -DWITH_TESTING=OFF -DON_INFER=ON
```
使用 TensorRT：
如果想使用 TensorRT 进行推理，首先需要下载并解压[TensorRT](https://developer.nvidia.com/tensorrt), 
在 cmake 中开启 WITH_TENSORRT， 并通过 TENSORRT_ROOT 指定刚刚解压的 TensorRT 路径。假设下载的 TensorRT lib 解压
到 D 盘目录下， cmake 编译指令如下：
```shell
cmake .. -GNinja -DWITH_GPU=ON  -DCMAKE_BUILD_TYPE=Release -DWITH_UNITY_BUILD=ON -DWITH_TESTING=OFF -DON_INFER=ON -DWITH_TENSORRT=ON -DTENSORRT_ROOT="D:/TensorRT"
```

其他编译选项含义请参见编译选项表。

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

注意：

	如果本机安装了多个 CUDA，将使用最新安装的 CUDA 版本，且无法指定。
	如果本机安装了多个 Python，将使用最新安装的 Python 版本。若需要指定 Python 版本，则需要指定 Python 路径，例如：

	cmake .. -GNinja -DWITH_GPU=ON -DPYTHON_EXECUTABLE=C:\Python38\python.exe -DPYTHON_INCLUDE_DIR=C:\Python38\include -DPYTHON_LIBRARY=C:\Python38\libs\python38.lib -DCMAKE_BUILD_TYPE=Release -DWITH_UNITY_BUILD=ON -DWITH_TESTING=OFF -DON_INFER=ON

**8. 执行编译：**
```shell
ninja -j4
```

**9. 编译成功后进入 python\dist 目录下找到生成的 .whl 包并安装：**
```shell
cd python\dist
pip3 install（whl 包的名字）--force-reinstall
```

**10. 在 paddle_inference_install_dir 目录下有生成的 paddle inference c++ 库和头文件等**

**11. 验证安装**
安装完成后你可以使用 python 进入 python 解释器，输入：
```shell
import paddle
paddle.utils.run_check()
```
如果出现 `PaddlePaddle is installed successfully!`，说明你已成功安装。


恭喜，至此你已完成 Paddle Inference 的编译安装