# Go 推理部署

本文主要介绍 Paddle Inferrence Go API 的安装。主要分为以下三个章节：环境准备，安装步骤，和验证安装。

## Paddle Inference 集成 Golang 的方式

Golang 为了能够尽可能的复用现有 C/C++ 库的软件，提供了 cgo 工具来作为 Golang 与 C/C++ 交互的方式。 Paddle Inference 提供了完善的 CAPI 接口，Golang 通过 cgo 直接接入，集成代码见[code](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/fluid/inference/goapi).

## 环境准备

Paddle Inference Go API 目前仅在 Linux 系统下 Golang 1.15 版本上进行了验证、测试和 CI 监控，如需其它环境和版本的支持，请在 issue 中描述需求，相关工作人员看到后会排期支持。

### Golang 安装

安装 Golang, 您可直接访问 [Golang 官网](https://go.dev/dl/)，下载对应版本的 Golang.

1. 下载 Golang 1.15 版本。

```bash
wget https://go.dev/dl/go1.15.15.linux-amd64.tar.gz
```

2. 卸载其它版本的 Golang.（可能需要 sudo 权限）

```
rm -rf /usr/local/go && tar -C /usr/local -xzf go1.15.15.linux-amd64.tar.gz
```

3. 修改 PATH 环境变量。

```
export PATH=$PATH:/usr/local/go/bin
```

4. 验证 Golang 正确安装。 打印出正确的版本号即代表安装成功。

```
go version
```

## 安装步骤

### Paddle Inference C 库安装

安装Paddle Inference C 库请参考[Paddle-Inference-C TODO待补充url]()。

Paddle Inference C 库的目录结构如下所示：

编译完成后，在编译目录下的 `paddle_inference_c_install_dir` 即为 C 预测库，目录结构如下：

```bash
paddle_inference_c_install_dir
├── paddle
│   ├── include               C 预测库头文件目录
│   │   └── pd_common.h
│   │   └── pd_config.h
│   │   └── pd_inference_api.h         C 预测库头文件
│   │   └── pd_predictor.h
│   │   └── pd_tensor.h
│   │   └── pd_types.h
│   │   └── pd_utils.h
│   └── lib
│       ├── libpaddle_inference_c.a          C 静态预测库文件
│       └── libpaddle_inference_c.so         C 动态预测库文件
├── third_party
│   └── install                          第三方链接库和头文件
│       ├── cryptopp
│       ├── gflags
│       ├── glog
│       ├── mkldnn
│       ├── mklml
│       ├── protobuf
│       └── xxhash
└── version.txt                          版本信息与编译选项信息
```

其中 `version.txt` 文件中记录了该预测库的版本信息，包括Git Commit ID、使用OpenBlas或MKL数学库、CUDA/CUDNN版本号，如：

```bash
GIT COMMIT ID: 47fa64004362b1d7d63048016911e62dc1d84f45
WITH_MKL: ON
WITH_MKLDNN: ON
WITH_GPU: ON
WITH_ROCM: OFF
WITH_ASCEND_CL: OFF
WITH_ASCEND_CXX11: OFF
WITH_IPU: OFF
CUDA version: 11.2
CUDNN version: v8.2
CXX compiler version: 8.2.0
WITH_TENSORRT: ON
TensorRT version: v8.2.4.2
```

此处需要记录 git commit id, 请替换该变量为您 C 库的 commit id.
```
export COMMITID=47fa64004362b1d7d63048016911e62dc1d84f45
```

### Paddle Inference Golang API 安装

1. 确认使用 Paddle 的 CommitId. 安装 C 库的过程中，记录下使用 Paddle 的 CommitId.

2. 使用 go get 获取 golang paddle api.

```
go env -w GO111MODULE=on
go get -d -v github.com/paddlepaddle/paddle/paddle/fluid/inference/goapi@${COMMITID}
```

3. 软链 C 库

go1.15 新增了 GOMODCACHE 环境变量，go get 默认会将代码下载到 GOMODCACHE 目录下，您可以通过 `go env | grep GOMODCACHE` 的方式，查看该路径，在官网发布的docker镜像中该路径一般默认为 `/root/gopath/pkg/mod`，进入到 golang api 代码路径建立软连接，将 C 预测库命名为 `paddle_inference_c`.

```bash
eval $(go env | grep GOMODCACHE)
# 按需修改最后的goapi版本号
cd ${GOMODCACHE}/github.com/paddlepaddle/paddle/paddle/fluid/inference/goapi\@v0.0.0-20210623023452-0722297d9b8c/
ln -s ${PADDLE_C_DOWNLOAD_DIR}/paddle_inference_c_install_dir paddle_inference_c
```

## 验证安装

在 Golang API 安装目录下存在`test.sh`脚本，用来检验安装是否成功，直接运行该脚本即可。

```bash
# 按需修改最后的goapi版本号
cd ${GOMODCACHE}/github.com/paddlepaddle/paddle/paddle/fluid/inference/goapi\@v0.0.0-20210623023452-0722297d9b8c/

bash test.sh
```

# Demo 示例

Golang Demo 示例见 [Paddle-Inference-Demo](https://github.com/PaddlePaddle/Paddle-Inference-Demo/tree/master/go)

