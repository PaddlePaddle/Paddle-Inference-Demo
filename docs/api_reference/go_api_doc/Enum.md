# 枚举类型

## DataType

DataType 为模型中 Tensor 的数据精度，默认值为 `FLOAT32`。枚举变量与 API 定义如下：

```go
// DataType 枚举类型定义
enum DataType { FLOAT32, INT32, INT64, UINT8, UNKDTYPE };

// 获取输入 dtype 的数据大小
// 参数：dtype - PaddleDType 枚举类型
// 返回：int32 - dtype 对应的数据大小
func SizeofDataType(dtype PaddleDType) int32
```

代码示例：

```go
package main

// 引入 Paddle Golang Package
import "/pathto/Paddle/go/paddle"

func main() {
    println("FLOAT32 size is: ", paddle.SizeofDataType(paddle.FLOAT32)); // 4
    println("INT64 size is: ", paddle.SizeofDataType(paddle.INT64)); // 8
    println("INT32 size is: ", paddle.SizeofDataType(paddle.INT32)); // 4
    println("UINT8 size is: ", paddle.SizeofDataType(paddle.UINT8)); // 1
}
```

## Precision

Precision 设置模型的运行精度，默认值为 `Precision_FLOAT32`。枚举变量定义如下：

```go
// PrecisionType 枚举类型定义
enum Precision { Precision_FLOAT32, Precision_INT8, Precision_HALF };
```

代码示例：

```go
package main

// 引入 Paddle Golang Package
import "/pathto/Paddle/go/paddle"

func main() {
    // 创建 AnalysisConfig 对象
    config := paddle.NewAnalysisConfig()
  
    // 启用 GPU 进行预测 - 初始化 GPU 显存 100M, Deivce_ID 为 0
    config.EnableUseGpu(100, 0)

    // 启用 TensorRT 进行预测加速 - FP32
    config.EnableTensorRtEngine(1 << 20, 1, 3, paddle.Precision_FLOAT32, false, false)

    // 启用 TensorRT 进行预测加速 - FP16
    config.EnableTensorRtEngine(1 << 20, 1, 3, paddle.Precision_HALF, false, false)

    // 启用 TensorRT 进行预测加速 - Int8
    config.EnableTensorRtEngine(1 << 20, 1, 3, paddle.Precision_INT8, false, false)
  
    // 通过 API 获取 TensorRT 启用结果 - true
    println("Enable TensorRT is: ", config.TensorrtEngineEnabled())
}
```
