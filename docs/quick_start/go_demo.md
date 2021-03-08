# 预测示例 (GO)

本章节包含2部分内容：(1) [运行 GO 示例程序](#id1)；(2) [GO 预测程序开发说明](#id7)。

## 运行 GO 示例程序

### 1. 源码编译 GO 预测库

Paddle Inference 的 GO 预测库即为 C 预测库，需要以源码编译的方式进行获取，请参照以下两个文档进行源码编译

- [安装与编译 Linux 预测库](../user_guides/source_compile.html#ubuntu-18-04)
- [安装与编译 Windows 预测库](../user_guides/source_compile.html#windows-10)

编译完成后，在编译目录下的 `paddle_inference_c_install_dir` 即为 GO 预测库，目录结构为：

```bash
paddle_inference_c_install_dir
├── paddle
│   ├── include
│   │   └── paddle_c_api.h               C/GO 预测库头文件
│   └── lib
│       ├── libpaddle_inference_c.a          C/GO 静态预测库文件
│       └── libpaddle_inference_c.so         C/GO 动态预测库文件
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
CUDA version: 9.0
CUDNN version: v7.6
CXX compiler version: 4.8.5
WITH_TENSORRT: ON
TensorRT version: v6
```

### 2. 准备预测部署模型

下载 [mobilenetv1](https://paddle-inference-dist.cdn.bcebos.com/mobilenet-test-model-data.tar.gz) 模型后解压，得到 Paddle Combined 形式的模型和数据，位于文件夹 data 下。可将 `__model__` 文件通过模型可视化工具 Netron 打开来查看模型结构。

```bash
wget https://paddle-inference-dist.cdn.bcebos.com/mobilenet-test-model-data.tar.gz
tar zxf mobilenet-test-model-data.tar.gz

# 获得 data 目录结构如下
data/
├── model
│   ├── __model__
│   └── __params__
├── data.txt
└── result.txt
```

### 3. 获取预测示例代码

本章节 GO 预测示例代码位于 [Paddle/go](https://github.com/PaddlePaddle/Paddle/tree/develop/go)，目录包含以下文件：

```bash
Paddle/go/
├── demo
│   ├── mobilenet_c.cc
│   ├── mobilenet_cxx.cc
│   └── mobilenet.go         GO 的预测示例程序源码
├── paddle
│   ├── common.go
│   ├── config.go            Config 的 GO references to C
│   ├── predictor.go         Predictor 的 GO references to C
│   ├── tensor.go            Tensor 的 GO references to C
└── README_cn.md             GO Demo README 说明
```

### 4. 准备预测执行目录

执行预测程序之前需要完成以下几个步骤

1. 将本章节 [第1步](#id2) 中的 `paddle_inference_c_install_dir` 移到到 `Paddle/go` 目录下，并将 `paddle_inference_c_install_dir` 重命名为 `paddle_c`
2. 本章节 [第2步](#id3) 中下载的模型和数据文件夹 `data` 移动到 `Paddle/go` 目录下

执行完之后的目录结构如下：

```bash
Paddle/go/
├── demo
│   ├── mobilenet_c.cc
│   ├── mobilenet_cxx.cc
│   └── mobilenet.go
├── paddle
│   ├── config.go                            Config 的 GO references to C
│   ├── predictor.go                         Predictor 的 GO references to C
│   ├── tensor.go                            Tensor 的 GO references to C
│   └── common.go
├── paddle_c                                 本章节第1步中的 paddle_inference_c_install_dir
│   ├── paddle
│   │   ├── include
│   │   │   └── paddle_c_api.h               C/GO 预测库头文件
│   │   └── lib
│   │       ├── libpaddle_inference_c.a          C/GO 静态预测库文件
│   │       └── libpaddle_inference_c.so         C/GO 动态预测库文件
│   └── third_party
├── data                                     本章节第2步中下载的模型和数据文件夹
│   ├── model
│   │   ├── __model__
│   │   └── __params__
│   ├── data.txt
│   └── result.txt
└── README_cn.md                             GO Demo README 说明
```

### 5. 执行预测程序

**注意**：需要先将动态库文件 `libpaddle_inference_c.so` 所在路径加入 `LD_LIBRARY_PATH`，否则会出现无法找到库文件的错误。

```bash
# 执行预测程序
export LD_LIBRARY_PATH=`pwd`/paddle_c/paddle/lib:$LD_LIBRARY_PATH
go run ./demo/mobilenet.go
```

成功执行之后，得到的预测输出结果如下：

```bash
# 程序输出结果如下
WARNING: Logging before InitGOogleLogging() is written to STDERR
I1211 11:46:11.061076 21893 pd_config.cc:43] data/model/__model__
I1211 11:46:11.061228 21893 pd_config.cc:48] data/model/__model__
W1211 11:46:11.061488 21893 analysis_predictor.cc:1052] Deprecated. Please use CreatePredictor instead.
============== paddle inference ==============
input num:  1
input name:  image
output num:  1
output name:  image
============== run inference =================
============= parse output ===================
v:  +3.000000e+000 +2.676507e-002 ...
137 6
137
```

## GO 预测程序开发说明

使用 Paddle Inference 开发 GO 预测程序仅需以下六个步骤：


(1) 引用 Paddle 的 GO references to C 目录

```go
import "/pathto/Paddle/go/paddle"
```

(2) 创建配置对象，并指定预测模型路径，详细可参考 [go API 文档 - Config](../api_reference/go_api_doc/Config_index)

```go
// 配置 PD_AnalysisConfig
config := paddle.NewAnalysisConfig()

// 设置预测模型路径，即为本小节第2步中下载的模型
config.SetModel("data/model/__model__", "data/model/__params__")
```

(3) 根据Config创建预测对象，详细可参考 [go API 文档 - Predictor](../api_reference/go_api_doc/Predictor)	

```go
predictor := paddle.NewPredictor(config)
```

(4) 设置模型输入和输出 Tensor，详细可参考 [go API 文档 - Tensor](../api_reference/go_api_doc/Tensor)

```go
// 创建输入 Tensor
input := predictor.GetInputTensors()[0]
output := predictor.GetOutputTensors()[0]

filename := "data/data.txt"
data := ReadData(filename)
input.SetValue(data[:1 * 3 * 300 * 300])
input.Reshape([]int32{1, 3, 300, 300})
```

(5) 执行预测引擎，详细可参考 [go API 文档 - Predictor](../api_reference/go_api_doc/Predictor)

```go
predictor.SetZeroCopyInput(input)
predictor.ZeroCopyRun()
predictor.GetZeroCopyOutput(output)
```

(6) 获得预测结果，详细可参考 [go API 文档 - Tensor](../api_reference/go_api_doc/Tensor)

```go
// 获取预测输出 Tensor 信息
output_val := output.Value()
value := reflect.ValueOf(output_val)
shape, dtype := paddle.ShapeAndTypeOf(value)

// 获取输出 float32 数据
v := value.Interface().([][]float32)
println("v: ", v[0][0], v[0][1], "...")
println(shape[0], shape[1])
println(output.Shape()[0])
```
