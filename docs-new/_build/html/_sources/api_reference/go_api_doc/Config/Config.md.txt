# 创建 Config

`Config` 对象相关方法用于创建预测相关配置，构建 `Predictor` 对象的配置信息，如模型路径、是否开启gpu等等。

相关方法定义如下：

```go
// 创建 Config 对象
// 参数：None
// 返回：*Config - Config 对象指针
func Config() *Config

// 判断当前 Config 是否有效
// 参数：None
// 返回：bool - 当前 Config 是否有效
func (config *Config) IsValid() bool
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

    // 判断当前 Config 是否有效 - true
    fmt.Println("Config validation is: ", config.IsValid())
}
```