# 使用 GPU 进行预测

**注意：**
1. Config 默认使用 CPU 进行预测，需要通过 `EnableUseGpu` 来启用 GPU 预测
2. 可以尝试启用 TensorRT 进行 GPU 预测加速

## GPU 设置

API定义如下：

```go
// 启用 GPU 进行预测
// 参数：memorySize - 初始化分配的 GPU 显存，以 MB 为单位
//      deviceId - 设备 id
// 返回：None
func (config *Config) EnableUseGpu(memorySize uint64, deviceId int32)

// 禁用 GPU 进行预测
// 参数：无
// 返回：None
func (config *Config) DisableGpu()

// 判断是否启用 GPU 
// 参数：无
// 返回：bool - 是否启用 GPU 
func (config *Config) UseGpu() bool

// 获取 GPU 的 device id
// 参数：无
// 返回：int -  GPU 的 device id
func (config *Config) GpuDeviceId() int32

// 获取 GPU 的初始显存大小
// 参数：无
// 返回：int -  GPU 的初始的显存大小
func (config *Config) MemoryPoolInitSizeMb() int32

// 初始化显存占总显存的百分比
// 参数：无
// 返回：float32 - 初始的显存占总显存的百分比
func (config *Config) FractionOfGpuMemoryForPool() float32
```

GPU设置代码示例：

```go
package main

// 引入 Paddle Golang Package
import pd "github.com/paddlepaddle/paddle/paddle/fluid/inference/goapi"
import fmt

func main() {
    // 创建 Config 对象
    config := pd.NewConfig()
  
    // 启用 GPU 进行预测 - 初始化 GPU 显存 100MB, DeivceID 为 0
    config.EnableUseGpu(100, 0)
  
    // 通过 API 获取 GPU 信息
    fmt.Println("Use GPU is: ", config.UseGpu()) // True
    fmt.Println("GPU deivce id is: ", config.GpuDeviceId())
    fmt.Println("GPU memory size is: ", config.MemoryPoolInitSizeMb())
    fmt.Println("GPU memory frac is: ", config.FractionOfGpuMemoryForPool())
  
    // 禁用 GPU 进行预测
    config.DisableGpu()
  
    // 通过 API 获取 GPU 信息 - False
    fmt.Println("Use GPU is: ", config.UseGpu())
}
```

## TensorRT 设置

**注意：** 启用 TensorRT 的前提为已经启用 GPU，否则启用 TensorRT 无法生效

更多 TensorRT 详细信息，请参考 [使用Paddle-TensorRT库预测](../../../optimize/paddle_trt)。

API定义如下：

```go
// 启用 TensorRT 进行预测加速
// 参数：workspaceSize     - 指定 TensorRT 使用的工作空间大小
//      maxBatchSize      - 设置最大的 batch 大小，运行时 batch 大小不得超过此限定值
//      minSubgraphSize   - Paddle-TRT 是以子图的形式运行，为了避免性能损失，当子图内部节点个数
//                          大于 min_subgraph_size 的时候，才会使用 Paddle-TRT 运行
//      precision         - 指定使用 TRT 的精度，支持 FP32(kFloat32)，FP16(kHalf)，Int8(kInt8)
//      useStatic         - 若指定为 true，在初次运行程序的时候会将 TRT 的优化信息进行序列化到磁盘上，
//                          下次运行时直接加载优化的序列化信息而不需要重新生成
//      useCalibMode      - 若要运行 Paddle-TRT INT8 离线量化校准，需要将此选项设置为 true
// 返回：None
func (config *Config) EnableTensorRtEngine(workspaceSize int64, maxBatchSize int32, minSubgraphSize int32,
	precision Precision, useStatic bool, useCalibMode bool)

// 设置 TensorRT 的动态 Shape
// 参数：minInputShape          - TensorRT 子图支持动态 shape 的最小 shape
//      maxInputShape          - TensorRT 子图支持动态 shape 的最大 shape
//      optimInputShape        - TensorRT 子图支持动态 shape 的最优 shape
//      disableTrtPluginFp16   - 设置 TensorRT 的 plugin 不在 fp16 精度下运行
// 返回：None
func (config *Config) SetTRTDynamicShapeInfo(minInputShape map[string][]int32, 
                                             maxInputShape map[string][]int32, 
                                             optimInputShape map[string][]int32, 
                                             disableTrtPluginFp16 bool)

// 判断是否启用 TensorRT 
// 参数：无
// 返回：bool - 是否启用 TensorRT
func (config *Config) TensorrtEngineEnabled() bool
```

代码示例：使用 TensorRT FP32 / FP16 / INT8 进行预测

```go
package main

// 引入 Paddle Golang Package
import pd "github.com/paddlepaddle/paddle/paddle/fluid/inference/goapi"
import fmt

func main() {
    // 创建 Config 对象
    config := pd.NewConfig()
  
    // 启用 GPU 进行预测 - 初始化 GPU 显存 100MB, Deivce_ID 为 0
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
