# 使用 CPU 进行预测

**注意：**
1. 在 CPU 型号允许的情况下，进行预测库下载或编译试尽量使用带 AVX 和 MKL 的版本
2. 可以尝试使用 Intel 的 MKLDNN 进行 CPU 预测加速，默认 CPU 不启用 MKLDNN
3. 在 CPU 可用核心数足够时，可以通过设置 `SetCpuMathLibraryNumThreads` 将线程数调高一些，默认线程数为 1

## CPU 设置

API定义如下：

```go
// 设置 CPU Blas 库计算线程数
// 参数：mathThreadsNum - blas库计算线程数
// 返回：None
func (config *Config) SetCpuMathLibraryNumThreads(mathThreadsNum int32)

// 获取 CPU Blas 库计算线程数
// 参数：无
// 返回：int - cpu blas 库计算线程数
func (config *Config) CpuMathLibraryNumThreads() int32
```

代码示例：

```go
package main

// 引入 Paddle Golang Package
import pd "github.com/paddlepaddle/paddle/paddle/fluid/inference/goapi"
import fmt

func main() {
    // 创建 Config 对象
    config := paddle.NewConfig()

    // 设置预测模型路径，这里为非 Combined 模型
    config.SetCpuMathLibraryNumThreads(10)

    // 输出模型路径
    fmt.Println("CPU Math Lib Thread Num is: ", config.CpuMathLibraryNumThreads())
}
```

## MKLDNN 设置

**注意：** 
1. 启用 MKLDNN 的前提为已经使用 CPU 进行预测，否则启用 MKLDNN 无法生效
2. 启用 MKLDNN BF16 要求 CPU 型号可以支持 AVX512，否则无法启用 MKLDNN BF16

API定义如下：

```go
// 启用 MKLDNN 进行预测加速
// 参数：无
// 返回：None
func (config *Config) EnableMkldnn()

// 判断是否启用 MKLDNN
// 参数：无
// 返回：bool - 是否启用 MKLDNN
func (config *Config) MkldnnEnabled() bool

// 启用 MKLDNN BFLOAT16
// 参数：无
// 返回：None
func (config *Config) EnableMkldnnBfloat16()

// 判断是否启用 MKLDNN BFLOAT16
// 参数：无
// 返回：bool - 是否启用 MKLDNN BFLOAT16
func (config *Config) MkldnnBfloat16Enabled() bool
```

代码示例：

```go
package main

// 引入 Paddle Golang Package
import pd "github.com/paddlepaddle/paddle/paddle/fluid/inference/goapi"
import fmt

func main() {
    // 创建 Config 对象
    config := pd.NewConfig()

    // 启用 MKLDNN 进行预测
    config.EnableMkldnn()

    // 通过 API 获取 MKLDNN 启用结果 - true
    fmt.Println("Enable MKLDNN is: ", config.MkldnnEnabled())

    // 启用 MKLDNN BFLOAT16 进行预测
    config.EnableMkldnnBfloat16()

    // 通过 API 获取 MKLDNN BFLOAT16 启用结果
    // 如果当前CPU支持AVX512，则返回 true, 否则返回 false
    fmt.Println("Enable MKLDNN BF16 is: ", config.MkldnnBfloat16Enabled())
}
```