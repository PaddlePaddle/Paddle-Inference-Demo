#  ZeroCopyTensor 方法

ZeroCopyTensor 是 Paddle Inference 的数据组织形式，用于对底层数据进行封装并提供接口对数据进行操作，包括设置 Shape、数据、LoD 信息等。

**注意：** 应使用 `Predictor` 的 `GetInputTensors` 和 `GetOutputTensors` 接口获取输入输出 `ZeroCopyTensor`。

ZeroCopyTensor 的API定义如下：

```go
// 获取 ZeroCopyTensor 维度信息
// 参数：tensor - ZeroCopyTensor 对象指针
// 返回：[]int32 - 包含 ZeroCopyTensor 维度信息的int数组
func (tensor *ZeroCopyTensor) Shape() []int32

// 设置 ZeroCopyTensor 维度信息
// 参数：tensor - ZeroCopyTensor 对象指针
//      shape - 包含维度信息的int数组
// 返回：None
func (tensor *ZeroCopyTensor) Reshape(shape []int32)

// 获取 ZeroCopyTensor 名称
// 参数：tensor - ZeroCopyTensor 对象指针
// 返回：string - ZeroCopyTensor 名称
func (tensor *ZeroCopyTensor) Name() string

// 设置 ZeroCopyTensor 名称
// 参数：tensor - ZeroCopyTensor 对象指针
//      name - ZeroCopyTensor 名称
// 返回：None
func (tensor *ZeroCopyTensor) Rename(name string)

// 获取 ZeroCopyTensor 数据类型
// 参数：tensor - ZeroCopyTensor 对象指针
// 返回：PaddleDType - ZeroCopyTensor 数据类型
func (tensor *ZeroCopyTensor) DataType() PaddleDType

// 设置 ZeroCopyTensor 数据
// 参数：tensor - ZeroCopyTensor 对象指针
//      value - ZeroCopyTensor 数据
// 返回：None
func (tensor *ZeroCopyTensor) SetValue(value interface{})

// 获取 ZeroCopyTensor 数据
// 参数：tensor - ZeroCopyTensor 对象指针
// 返回：interface{} - ZeroCopyTensor 数据
func (tensor *ZeroCopyTensor) Value() interface{}
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