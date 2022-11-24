# 快速上手GO推理

本章节包含2部分内容,
- [运行 GO 示例程序](#id1)
- [GO 推理程序开发说明](#id2)

注意本章节文档和代码仅适用于Linux系统。

## 运行 GO 示例程序

在此环节中，共包含以下5个步骤，
- 环境准备
- 模型准备
- 推理代码
- 编译代码
- 执行程序


### 1. 环境准备

go 语言推理需要下载Paddle Inference的 Go 预编译推理库。Paddle Inference 提供了 Ubuntu/Windows/MacOS 平台的官方 Release 推理库下载，用户需根据开发环境和硬件自行下载安装，具体可参阅 [Go 推理环境安装](../install/go_install.md)。
 
执行推理程序之前需要完成以下几个步骤

1. 使用`go get`获取golang paddle api，`go` 的版本需要大于等于 `1.15`

```
# 此处使用对应tag的CommitId，假设为76e5724，可在步骤1中查看到
export GO111MODULE=on
go get -d -v github.com/paddlepaddle/paddle/paddle/fluid/inference/goapi@590b4dbcdd989324089ce43c22ef151c746c92a3
```

2. 软链

`go get`默认会将代码下载到`GOMODCACHE`目录下，您可以通过`go env | grep GOMODCACHE`的方式，查看该路径，在官网发布的docker镜像中该路径一般默认为`/root/gopath/pkg/mod`，进入到golang api代码路径建立软连接，将c推理库命名为`paddle_inference_c`。

```bash
eval $(go env | grep GOMODCACHE)
# 按需修改最后的goapi版本号
cd ${GOMODCACHE}/github.com/paddlepaddle/paddle/paddle/fluid/inference/goapi\@v0.0.0-20220523104455-d5b6eec273a9/
ln -s ${PADDLE_C_DOWNLOAD_DIR}/paddle_inference_c paddle_inference_c
```

3. 进入到golang api代码路径后，运行单测，验证。

```
bash test.sh
```
单测通过后，即表示 go 环境准备完成。

### 2. 模型准备

下载 [resnet50](https://paddle-inference-dist.bj.bcebos.com/Paddle-Inference-Demo/resnet50.tgz) 模型后解压。

```bash
wget https://paddle-inference-dist.bj.bcebos.com/Paddle-Inference-Demo/resnet50.tgz
tar zxf resnet50.tgz

# 获得 resnet50 目录结构如下
resnet50/
├── inference.pdmodel
├── inference.pdiparams
└── inference.pdiparams.info
```

### 3. 推理代码

本章节 GO 推理示例代码位于 [Paddle-Inference-Demo](https://github.com/PaddlePaddle/Paddle-Inference-Demo/tree/master/go)，目录下的resnet50子目录。

```
# 获取部署 Demo 代码库
git clone https://github.com/PaddlePaddle/Paddle-Inference-Demo.git
cd Paddle-Inference-Demo/go/resnet50
```
其中示例代码目录结构如下所示
```
Paddle-Inference-Demo/go/resnet50/
├── README.md                README 说明
└── demo.go                  示例代码
```
### 4. 编译代码

进入`Paddle-Inference-Demo/go/resnet50/`目录，执行
```
# 先将依赖库加入环境变量
export LD_LIBRARY_PATH=${PADDLE_C_DOWNLOAD_DIR}/paddle_inference_c/third_party/install/paddle2onnx/lib/:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${PADDLE_C_DOWNLOAD_DIR}/paddle_inference_c/third_party/install/onnxruntime/lib/:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${PADDLE_C_DOWNLOAD_DIR}/paddle_inference_c/third_party/install/mklml/lib/:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${PADDLE_C_DOWNLOAD_DIR}/paddle_inference_c/third_party/install/mkldnn/lib/:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${PADDLE_C_DOWNLOAD_DIR}/paddle_inference_c/paddle/lib/:${LD_LIBRARY_PATH}

go mod init demo
go get -d -v github.com/paddlepaddle/paddle/paddle/fluid/inference/goapi@v0.0.0-20220523104455-d5b6eec273a9
go build .
```

### 5. 执行程序

在将模型`resnet50`拷贝至编译目录后，使用如下命令执行程序
```
./demo -thread_num 1 -work_num 1 -cpu_math 2
```

## GO 推理程序开发说明

使用 Paddle Inference 开发 GO 推理程序仅需以下六个步骤：


(1) 引用 Paddle Inference 的 GO API

```go
import pd "github.com/paddlepaddle/paddle/paddle/fluid/inference/goapi"
```

(2) 创建配置对象，并指定推理模型路径，详细可参考 [go API 文档 - Config](../../api_reference/go_api_doc/Config_index.rst)

```go
// 配置 PD_AnalysisConfig
config := paddle.NewConfig()

// 设置推理模型路径，即为本小节第2步中下载的模型
config.SetModel("resnet50/inference.pdmodel", "resnet50/inference.pdiparams")
```

(3) 根据Config创建推理对象，详细可参考 [go API 文档 - Predictor](../../api_reference/go_api_doc/Predictor.md)	

```go
predictor := paddle.NewPredictor(config)
```

(4) 设置模型输入和输出 Tensor，详细可参考 [go API 文档 - Tensor](../../api_reference/go_api_doc/Tensor.md)

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

(5) 执行推理引擎，详细可参考 [go API 文档 - Predictor](../../api_reference/go_api_doc/Predictor.md)

```go
predictor.Run()
```

(6) 获得推理结果，详细可参考 [go API 文档 - Tensor](../../api_reference/go_api_doc/Tensor.md)

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
