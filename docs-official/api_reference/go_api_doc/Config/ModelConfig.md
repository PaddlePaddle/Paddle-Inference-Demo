# 设置预测模型

## 从文件中加载预测模型

API定义如下：

```go
// 设置模型文件路径
// 参数：model - 模型文件所在路径
//      params - 模型参数文件所在路径
// 返回：None
func (config *Config) SetModel(model, params string)

// 获取 Combined 模型的模型文件路径
// 参数：无
// 返回：string - 模型文件路径
func (config *Config) ProgFile() string

// 获取 Combined 模型的参数文件路径
// 参数：无
// 返回：string - 参数文件路径
func (config *Config) ParamsFile() string
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

    // 设置预测模型路径，这里为 Combined 模型
    config.SetModel("data/resnet.pdmodel", "data/resnet.pdiparams")

    // 输出模型路径
    fmt.Println("Combined model path is: ", config.ProgFile())
    fmt.Println("Combined param path is: ", config.ParamsFile())
}
```
