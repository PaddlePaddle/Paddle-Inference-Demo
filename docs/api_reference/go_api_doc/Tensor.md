#  Tensor 方法

Tensor 是 Paddle Inference 的数据组织形式，用于对底层数据进行封装并提供接口对数据进行操作，包括设置 Shape、数据、LoD 信息等。

**注意：** 应使用 `Predictor` 的 `GetInputHandle` 和 `GetOutputHandle` 接口获取输入输出 `Tensor`。

Tensor 的API定义如下：

```go
// 获取 Tensor 维度信息
// 参数：无
// 返回：[]int32 - 包含 Tensor 维度信息的int数组
func (tensor *Tensor) Shape() []int32

// 设置 Tensor 维度信息
// 参数：shape - 包含维度信息的int数组
// 返回：None
func (tensor *Tensor) Reshape(shape []int32)

// 获取 Tensor 名称
// 参数：无
// 返回：string - Tensor 名称
func (tensor *Tensor) Name() string

// 获取 Tensor 数据类型
// 参数：无
// 返回：DataType - Tensor 数据类型
func (tensor *Tensor) Type() DataType

// 设置 Tensor 数据
// 参数：value - Tensor 数据
// 返回：None
func (tensor *Tensor) CopyFromCpu(value interface{})

// 获取 Tensor 数据
// 参数：value - 用来存储 Tensor 数据
// 返回：None
func (t *Tensor) CopyToCpu(value interface{})
```

代码示例：

```go
package main

// 引入 Paddle Golang Package
import pd "github.com/paddlepaddle/paddle/paddle/fluid/inference/goapi"
import fmt

func main() {
    // 创建 AnalysisConfig 对象
    config := pd.NewConfig()

    // 设置预测模型路径，这里为非 Combined 模型
    config.SetModel("data/mobilenet_v1", "")
    // config.SetModel("data/model/__model__", "data/model/__params__")

    // 根据 Config 构建预测执行对象 Predictor
    predictor := pd.NewPredictor(config)

    // 获取输入输出 Tensor 信息
    println("input num: ", predictor.GetInputNum())
    println("input name: ", predictor.GetInputNames()[0])
    println("output num: ", predictor.GetOutputNum())
    println("output name: ", predictor.GetOutputNames()[0])

    // 获取输入输出 Tensor 指针
    input := predictor.GetInputHandle(predictor.GetInputNames()[0])
    output := predictor.GetOutputHandle(predictor.GetOutputNames()[0])

    inputData := make([]float32, 1 * 3 * 224 * 224)
    for i := 0; i < 1 * 3 * 224 * 224; i++ {
      inputData[i] = 1.0
    }

    // 设置输入 Tensor
    input.Reshape([]int32{1, 3, 224, 224})
    input.CopyFromCpu(inputData)

    // 执行预测
    predictor.Run()

    // 获取输出 Tensor
    outData := make([]float32, numElements(output.Shape()))
    output.CopyToCpu(outData)
}

func numElements(shape []int32) int32 {
	n := int32(1)
	for _, v := range shape {
		n *= v
	}
	return n
}
```
