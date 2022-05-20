
# 使用 ONNXRuntime 进行预测

API定义如下：

```go
// 启用 ONNXRuntime 进行预测
// 参数：None
// 返回：None
func (config *Config) EnableONNXRuntime()

// 禁用 ONNXRuntime 进行预测
// 参数：None
// 返回：None
func (config *Config) DisableONNXRuntime();

// 判断是否启用 ONNXRuntime 
// 参数：None
// 返回：bool - 是否启用 ONNXRuntime 
func (config *Config) ONNXRuntimeEnabled() bool;

// 启用 ONNXRuntime 预测时开启优化
// 参数：None
// 返回：None
func (config *Config) EnableORTOptimization();
```

ONNXRuntime设置代码示例：

```go
package main

// 引入 Paddle Golang Package
import pd "github.com/paddlepaddle/paddle/paddle/fluid/inference/goapi"
import fmt

func main() {
    // 创建 Config 对象
    config := pd.NewConfig()

    // 启用 ONNXRuntime 进行预测
    config.EnableONNXRuntime()
  
    // 通过 API 获取 ONNXRuntime 信息
    fmt.Println("Use ONNXRuntime is: ", config.ONNXRuntimeEnabled()) // True
  
    // 开启ONNXRuntime优化
    config.EnableORTOptimization();

    // 禁用 ONNXRuntime 进行预测
    config.DisableONNXRuntime()
  
    // 通过 API 获取 ONNXRuntime 信息
    fmt.Println("Use ONNXRuntime is: ", config.ONNXRuntimeEnabled()) // False
}
```
