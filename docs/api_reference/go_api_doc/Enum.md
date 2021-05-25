# 枚举类型

## DataType

DataType 为模型中 Tensor 的数据精度。变量定义如下：

```go
type DataType C.PD_DataType

const (
	Unk     DataType = C.PD_DATA_UNK
	Float32 DataType = C.PD_DATA_FLOAT32
	Int32   DataType = C.PD_DATA_INT32
	Int64   DataType = C.PD_DATA_INT64
	Uint8   DataType = C.PD_DATA_UINT8
	Int8    DataType = C.PD_DATA_INT8
)
```

## Precision

Precision 设置模型的运行精度。变量定义如下：

```go
type Precision C.PD_PrecisionType

const (
	PrecisionFloat32 Precision = C.PD_PRECISION_FLOAT32
	PrecisionInt8    Precision = C.PD_PRECISION_INT8
	PrecisionHalf    Precision = C.PD_PRECISION_HALF
)
```

代码示例：

```go
package main

// 引入 Paddle Golang Package
import pd "github.com/paddlepaddle/paddle/paddle/fluid/inference/goapi"
import fmt

func main() {
    // 创建 AnalysisConfig 对象
    config := pd.NewAnalysisConfig()
  
    // 启用 GPU 进行预测 - 初始化 GPU 显存 100M, Deivce_ID 为 0
    config.EnableUseGpu(100, 0)

    // 启用 TensorRT 进行预测加速 - FP32
    config.EnableTensorRtEngine(1 << 20, 1, 3, pd.PrecisionFloat32, false, false)

    // 启用 TensorRT 进行预测加速 - FP16
    config.EnableTensorRtEngine(1 << 20, 1, 3, pd.PrecisionHalf, false, false)

    // 启用 TensorRT 进行预测加速 - Int8
    config.EnableTensorRtEngine(1 << 20, 1, 3, pd.PrecisionInt8, false, false)
  
    // 通过 API 获取 TensorRT 启用结果 - true
    fmt.Println("Enable TensorRT is: ", config.TensorrtEngineEnabled())
}
```
