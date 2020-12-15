# 设置预测模型

## 从文件中加载预测模型 - 非Combined模型 

API定义如下：

```go
// 设置模型文件路径
// 参数：config - *AnalysisConfig 对象指针
//      model - 模型文件夹路径
//      params - "", 当输入模型为非 Combined 模型时，该参数为空字符串
// 返回：None
func (config *AnalysisConfig) SetModel(model, params string)

// 获取非combine模型的文件夹路径
// 参数：config - *AnalysisConfig 对象指针
// 返回：string - 模型文件夹路径
func (config *AnalysisConfig) ModelDir() string 
```

代码示例：

```go
package main

// 引入 Paddle Golang Package
import "/pathto/Paddle/go/paddle"

func main() {
    // 创建 AnalysisConfig 对象
    config := paddle.NewAnalysisConfig()

    // 设置预测模型路径，这里为非 Combined 模型
    config.SetModel("data/mobilenet_v1", "")

    // 输出模型路径
    println("Non-combined model dir is: ", config.ModelDir())
}
```

## 从文件中加载预测模型 -    Combined 模型

API定义如下：

```go
// 设置模型文件路径
// 参数：config - *AnalysisConfig 对象指针
//      model_dir - Combined 模型文件所在路径
//      params_path - Combined 模型参数文件所在路径
// 返回：None
func (config *AnalysisConfig) SetModel(model, params string)

// 获取 Combined 模型的模型文件路径
// 参数：config - *AnalysisConfig 对象指针
// 返回：string - 模型文件路径
func (config *AnalysisConfig) ProgFile() string

// 获取 Combined 模型的参数文件路径
// 参数：config - *AnalysisConfig 对象指针
// 返回：string - 参数文件路径
func (config *AnalysisConfig) ParamsFile() string
```

代码示例：

```go
package main

// 引入 Paddle Golang Package
import "/pathto/Paddle/go/paddle"

func main() {
    // 创建 AnalysisConfig 对象
    config := paddle.NewAnalysisConfig()

    // 设置预测模型路径，这里为 Combined 模型
    config.SetModel("data/model/__model__", "data/model/__params__")

    // 输出模型路径
    println("Combined model path is: ", config.ProgFile())
    println("Combined param path is: ", config.ParamsFile())
}
```
