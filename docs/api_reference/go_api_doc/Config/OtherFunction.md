# 启用内存优化

API定义如下：

```go
// 开启内存/显存复用，具体降低内存效果取决于模型结构
// 参数：无
// 返回：None
func (config *Config) EnableMemoryOptim()

// 判断是否开启内存/显存复用
// 参数：无
// 返回：bool - 是否开启内/显存复用
func (config *Config) MemoryOptimEnabled() bool
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

    // 开启 CPU 内存优化
    config.EnableMemoryOptim();
    // 通过 API 获取 CPU 是否已经开启显存优化 - true
    fmt.Println("CPU Mem Optim is: ", config.MemoryOptimEnabled())

    // 启用 GPU 进行预测
    config.EnableUseGpu(100, 0)
    // 开启 GPU 显存优化
    config.EnableMemoryOptim();
    // 通过 API 获取 GPU 是否已经开启显存优化 - true
    fmt.Println("GPU Mem Optim is: ", config.MemoryOptimEnabled())
}
```

# Profile 设置

API定义如下：

```go
// 打开 Profile，运行结束后会打印所有 OP 的耗时占比。
// 参数：无
// 返回：None
func (config *Config) EnableProfile()

// 判断是否开启 Profile
// 参数：无
// 返回：bool - 是否开启 Profile
func (config *Config) ProfileEnabled() bool
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

    // 打开 Profile
    config.EnableProfile();
    // 判断是否开启 Profile - true
    fmt.Println("Profile is: ", config.ProfileEnabled())
}
```

执行预测之后输出的 Profile 的结果如下：

```bash
------------------------->     Profiling Report     <-------------------------

Place: CPU
Time unit: ms
Sorted by total time in descending order in the same thread

-------------------------     Overhead Summary      -------------------------

Total time: 1085.33
  Computation time       Total: 1066.24     Ratio: 98.2411%
  Framework overhead     Total: 19.0902     Ratio: 1.75893%

-------------------------     GpuMemCpy Summary     -------------------------

GpuMemcpy                Calls: 0           Total: 0           Ratio: 0%

-------------------------       Event Summary       -------------------------

Event                            Calls       Total       Min.        Max.        Ave.        Ratio.
thread0::conv2d                  210         319.734     0.815591    6.51648     1.52254     0.294595
thread0::load                    137         284.596     0.114216    258.715     2.07735     0.26222
thread0::depthwise_conv2d        195         266.241     0.955945    2.47858     1.36534     0.245308
thread0::elementwise_add         210         122.969     0.133106    2.15806     0.585568    0.113301
thread0::relu                    405         56.1807     0.021081    0.585079    0.138718    0.0517635
thread0::batch_norm              195         25.8073     0.044304    0.33896     0.132345    0.0237783
thread0::fc                      15          7.13856     0.451674    0.714895    0.475904    0.0065773
thread0::pool2d                  15          1.48296     0.09054     0.145702    0.0988637   0.00136636
thread0::softmax                 15          0.941837    0.032175    0.460156    0.0627891   0.000867786
thread0::scale                   15          0.240771    0.013394    0.030727    0.0160514   0.000221841
```

# Log 设置

API定义如下：

```go
// 去除 Paddle Inference 运行中的 LOG
// 参数：无
// 返回：None
func (config *Config) DisableGlogInfo()
```

代码示例：

```go
package main

// 引入 Paddle Golang Package
import pd "github.com/paddlepaddle/paddle/paddle/fluid/inference/goapi"

func main() {
    // 创建 Config 对象
    config := paddle.NewConfig()

    // 去除 Paddle Inference 运行中的 LOG
    config.DisableGlogInfo();
}
```