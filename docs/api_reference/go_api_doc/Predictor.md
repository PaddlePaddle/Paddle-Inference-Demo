# Predictor 方法

Paddle Inference 的预测器，由 `NewPredictor` 根据 `AnalysisConfig` 进行创建。用户可以根据 Predictor 提供的接口设置输入数据、执行模型预测、获取输出等。

## 创建 Predictor

API定义如下：

```go
// 根据 Config 构建预测执行对象 Predictor
// 参数: config - 用于构建 Predictor 的配置信息
// 返回: *Predictor - 预测对象指针
func NewPredictor(config *AnalysisConfig) *Predictor

// 删除 Predictor 对象
// predictor - Predictor 对象指针
// 返回：None
func DeletePredictor(predictor *Predictor)
```

代码示例:

```go
package main

// 引入 Paddle Golang Package
import "/pathto/Paddle/go/paddle"

func main() {
    // 创建 AnalysisConfig 对象
    config := paddle.NewAnalysisConfig()

    // 设置预测模型路径，这里为非 Combined 模型
    config.SetModel("data/mobilenet_v1", "")

    // 根据 Config 构建预测执行对象 Predictor
    predictor := paddle.NewPredictor(config)

    // 删除 Predictor 对象
    paddle.DeletePredictor(predictor)
}
```

## 输入输出与执行预测

API 定义如下：

```go
// 获取模型输入 Tensor 的数量
// 参数：predictor - PD_Predictor 对象指针
// 返回：int - 模型输入 Tensor 的数量
func (predictor *Predictor) GetInputNum() int

// 获取模型输出 Tensor 的数量
// 参数：predictor - PD_Predictor 对象指针
// 返回：int - 模型输出 Tensor 的数量
func (predictor *Predictor) GetOutputNum() int 

// 获取输入 Tensor 名称
// 参数：predictor - PD_Predictor 对象指针
//      int - 输入 Tensor 的index
// 返回：string - 输入 Tensor 名称
func (predictor *Predictor) GetInputName(n int) string 

// 获取输出 Tensor 名称
// 参数：predictor - PD_Predictor 对象指针
//      int - 输出 Tensor 的index
// 返回：string - 输出 Tensor 名称
func (predictor *Predictor) GetOutputName(n int) string

// 获取输入 Tensor 指针
// 参数：predictor - PD_Predictor 对象指针
// 返回：*ZeroCopyTensor - 输入 Tensor 指针
func (predictor *Predictor) GetInputTensors() [](*ZeroCopyTensor)

// 获取输出 Tensor 指针
// 参数：predictor - PD_Predictor 对象指针
// 返回：*ZeroCopyTensor - 输出 Tensor 指针
func (predictor *Predictor) GetOutputTensors() [](*ZeroCopyTensor)

// 获取输入 Tensor 名称数组
// 参数：predictor - PD_Predictor 对象指针
// 返回：[]string - 输入 Tensor 名称数组
func (predictor *Predictor) GetInputNames() []string 

// 获取输出 Tensor 名称数组
// 参数：predictor - PD_Predictor 对象指针
// 返回：[]string - 输出 Tensor 名称数组
func (predictor *Predictor) GetOutputNames() []string

// 设置输入 Tensor
// 参数：predictor - PD_Predictor 对象指针
//      *ZeroCopyTensor - 输入 Tensor 指针
// 返回：None
func (predictor *Predictor) SetZeroCopyInput(tensor *ZeroCopyTensor)

// 获取输出 Tensor
// 参数：predictor - PD_Predictor 对象指针
//      *ZeroCopyTensor - 输出 Tensor 指针
// 返回：None
func (predictor *Predictor) GetZeroCopyOutput(tensor *ZeroCopyTensor)

// 执行预测
// 参数：predictor - PD_Predictor 对象指针
// 返回：None
func (predictor *Predictor) ZeroCopyRun()
```

代码示例：

```go
package main

// 引入 Paddle Golang Package
import "/pathto/Paddle/go/paddle"
import "reflect"

func main() {
    // 创建 AnalysisConfig 对象
    config := paddle.NewAnalysisConfig()
    config.SwitchUseFeedFetchOps(false)

    // 设置预测模型路径，这里为非 Combined 模型
    config.SetModel("data/mobilenet_v1", "")
    // config.SetModel("data/model/__model__", "data/model/__params__")

    // 根据 Config 构建预测执行对象 Predictor
    predictor := paddle.NewPredictor(config)

    // 获取输入输出 Tensor 信息
    println("input num: ", predictor.GetInputNum())
    println("input name: ", predictor.GetInputNames()[0])
    println("output num: ", predictor.GetOutputNum())
    println("output name: ", predictor.GetInputNames()[0])

    // 获取输入输出 Tensor 指针
    input := predictor.GetInputTensors()[0]
    output := predictor.GetOutputTensors()[0]

    input_data := make([]float32, 1 * 3 * 224 * 224)
    for i := 0; i < 1 * 3 * 224 * 224; i++ {
      input_data[i] = 1.0
    }
    input.SetValue(input_data)
    input.Reshape([]int32{1, 3, 224, 224})

    // 设置输入 Tensor
    predictor.SetZeroCopyInput(input)
    // 执行预测
    predictor.ZeroCopyRun()
    // 获取输出 Tensor
    predictor.GetZeroCopyOutput(output)

    // 获取输出 Tensor 信息
    output_val := output.Value()
    value := reflect.ValueOf(output_val)
    shape, dtype := paddle.ShapeAndTypeOf(value)
    v := value.Interface().([][]float32)
    println("Ouptut Shape is: ", shape[0], "x", shape[1])
    println("Ouptut Dtype is: ", dtype)
    println("Output Data is: ", v[0][0], v[0][1], v[0][2], v[0][3], v[0][4], "...")
    
    // 删除 Predictor 对象
    paddle.DeletePredictor(predictor)
}
```
