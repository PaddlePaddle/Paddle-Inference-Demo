# 预测示例 (GO)

本章节包含2部分内容：(1) [运行 GO 示例程序](#id1)；(2) [GO 预测程序开发说明](#id8)。

## 运行 GO 示例程序

### 1. 获取 C 预测库

#### 下载C预测库

您可以选择直接下载[paddle_inference_c预测库](../user_guides/download_lib.md)。

#### 源码编译方式获取 C 预测库

您可以源码编译的方式获取 C 预测库，请参照以下两个文档进行源码编译

- [安装与编译 Linux 预测库](../user_guides/source_compile.html#ubuntu-18-04)
- [安装与编译 Windows 预测库](../user_guides/source_compile.html#windows-10)

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
GIT COMMIT ID: 1bf4836580951b6fd50495339a7a75b77bf539f6
WITH_MKL: ON
WITH_MKLDNN: ON
WITH_GPU: ON
CUDA version: 10.1
CUDNN version: v7.6
CXX compiler version: 8.2
WITH_TENSORRT: ON
TensorRT version: v6
```

### 2. 准备预测部署模型

下载 [resnet50](https://paddle-inference-dist.bj.bcebos.com/Paddle-Inference-Demo/resnet50.tgz) 模型后解压，得到 Paddle Combined 形式的模型。

```bash
wget https://paddle-inference-dist.bj.bcebos.com/Paddle-Inference-Demo/resnet50.tgz
tar zxf resnet50.tgz

# 获得 resnet50 目录结构如下
resnet50/
├── inference.pdmodel
├── inference.pdiparams
└── inference.pdiparams.info
```

### 3. 获取预测示例代码

本章节 GO 预测示例代码位于 [Paddle-Inference-Demo](https://github.com/PaddlePaddle/Paddle-Inference-Demo/tree/master/go)，目录下的resnet50子目录。

### 4. 准备预测执行目录

执行预测程序之前需要完成以下几个步骤

1. 使用`go get`获取golang paddle api

```
# 此处使用对应tag的CommitId，假设为76e5724，可在步骤1中查看到
go get -d -v github.com/paddlepaddle/paddle/paddle/fluid/inference/goapi@76e5724
```

2. 软链

`go get`默认会将代码下载到`GOMODCACHE`目录下，您可以通过`go env | grep GOMODCACHE`的方式，查看该路径，在官网发布的docker镜像中该路径一般默认为`/root/gopath/pkg/mod`，进入到golang api代码路径建立软连接，将c预测库命名为`paddle_inference_c`。

```bash
eval $(go env | grep GOMODCACHE)
# 按需修改最后的goapi版本号
cd ${GOMODCACHE}/github.com/paddlepaddle/paddle/paddle/fluid/inference/goapi\@v0.0.0-20210517084506-76e5724c16a5/
ln -s ${PADDLE_C_DOWNLOAD_DIR}/paddle_inference_c_install_dir paddle_inference_c
```

3. 进入到golang api代码路径后，运行单测，验证。

```
bash test.sh
```

### 5. 编译执行

进入步骤3中预测示例代码所在目录，执行：

```
go mod init demo
go get -d -v github.com/paddlepaddle/paddle/paddle/fluid/inference/goapi@76e5724
go build .

./demo -thread_num 1 -work_num 1 -cpu_math 2
```

## GO 预测程序开发说明

使用 Paddle Inference 开发 GO 预测程序仅需以下六个步骤：


(1) 引用 Paddle Inference 的 GO API

```go
import pd "github.com/paddlepaddle/paddle/paddle/fluid/inference/goapi"
```

(2) 创建配置对象，并指定预测模型路径，详细可参考 [go API 文档 - Config](../api_reference/go_api_doc/Config_index)

```go
// 配置 PD_AnalysisConfig
config := paddle.NewConfig()

// 设置预测模型路径，即为本小节第2步中下载的模型
config.SetModel("resnet50/inference.pdmodel", "resnet50/inference.pdiparams")
```

(3) 根据Config创建预测对象，详细可参考 [go API 文档 - Predictor](../api_reference/go_api_doc/Predictor)	

```go
predictor := paddle.NewPredictor(config)
```

(4) 设置模型输入和输出 Tensor，详细可参考 [go API 文档 - Tensor](../api_reference/go_api_doc/Tensor)

```go
// 创建输入 Tensor
inNames := predictor.GetInputNames()
inHandle := predictor.GetInputHandle(inNames[0])

data := make([]float32, 1*3*224*224)
for i := 0; i < len(data); i++ {
    data[i] = float32(i%255) * 0.1
}
inHandle.Reshape([]int32{1, 3, 224, 224})
inHandle.CopyFromCpu(data)
```

(5) 执行预测引擎，详细可参考 [go API 文档 - Predictor](../api_reference/go_api_doc/Predictor)

```go
predictor.Run()
```

(6) 获得预测结果，详细可参考 [go API 文档 - Tensor](../api_reference/go_api_doc/Tensor)

```go
outNames := predictor.GetOutputNames()
outHandle := predictor.GetOutputHandle(outNames[0])
outData := make([]float32, numElements(outHandle.Shape()))
outHandle.CopyToCpu(outData)

func numElements(shape []int32) int32 {
	n := int32(1)
	for _, v := range shape {
		n *= v
	}
	return n
}
```
