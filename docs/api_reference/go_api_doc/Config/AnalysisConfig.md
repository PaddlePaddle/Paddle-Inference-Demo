# 创建 AnalysisConfig

`AnalysisConfig` 对象相关方法用于创建预测相关配置，构建 `Predictor` 对象的配置信息，如模型路径、是否开启gpu等等。

相关方法定义如下：

```go
// 创建 AnalysisConfig 对象
// 参数：None
// 返回：*AnalysisConfig - AnalysisConfig 对象指针
func NewAnalysisConfig() *AnalysisConfig

// 设置 AnalysisConfig 为无效状态，保证每一个 AnalysisConfig 仅用来初始化一次 Predictor
// 参数：config - *AnalysisConfig 对象指针
// 返回：None
func (config *AnalysisConfig) SetInValid()

// 判断当前 AnalysisConfig 是否有效
// 参数：config - *AnalysisConfig 对象指针
// 返回：bool - 当前 AnalysisConfig 是否有效
func (config *AnalysisConfig) IsValid() bool
```

代码示例：

```go
package main

// 引入 Paddle Golang Package
import "/pathto/Paddle/go/paddle"

func main() {
    // 创建 AnalysisConfig 对象
    config := paddle.NewAnalysisConfig()

    // 判断当前 Config 是否有效 - true
    println("Config validation is: ", config.IsValid())

    // 设置 Config 为无效状态
    config.SetInValid();

    // 判断当前 Config 是否有效 - false
    println("Config validation is: ", config.IsValid())
}
```