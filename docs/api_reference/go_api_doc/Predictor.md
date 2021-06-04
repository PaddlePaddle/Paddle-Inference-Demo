# Predictor 方法

Paddle Inference 的预测器，由 `NewPredictor` 根据 `Config` 进行创建。用户可以根据 Predictor 提供的接口设置输入数据、执行模型预测、获取输出等。

## 创建 Predictor

API定义如下：

```go
// 根据 Config 构建预测执行对象 Predictor
// 参数: config - 用于构建 Predictor 的配置信息
// 返回: *Predictor - 预测对象指针
func NewPredictor(config *Config) *Predictor
```

代码示例:

```go
package main

// 引入 Paddle Golang Package
import pd "github.com/paddlepaddle/paddle/paddle/fluid/inference/goapi"
import fmt

func main() {
    // 创建 Config 对象
    config := pd.NewConfig()

    // 设置预测模型路径，这里为非 Combined 模型
    config.SetModelDir("data/mobilenet_v1")

    // 根据 Config 构建预测执行对象 Predictor
    predictor := pd.NewPredictor(config)

    fmt.Printf("%T\n", predictor)
}
```

## 输入输出与执行预测

API 定义如下：

```go
// 获取模型输入 Tensor 的数量
// 参数：无
// 返回：uint - 模型输入 Tensor 的数量
func (p *Predictor) GetInputNum() uint

// 获取模型输出 Tensor 的数量
// 参数：无
// 返回：uint - 模型输出 Tensor 的数量
func (p *Predictor) GetOutputNum() uint

// 获取输入 Tensor 名称
// 参数：无
// 返回：[]string - 输入 Tensor 名称
func (p *Predictor) GetInputNames() []string

// 获取输出 Tensor 名称
// 参数：无
// 返回：[]string - 输出 Tensor 名称
func (p *Predictor) GetOutputNames() []string

// 获取输入 handle
// 参数：name - 输入handle名称
// 返回：*Tensor - 输入 handle
func (p *Predictor) GetInputHandle(name string) *Tensor

// 获取输出 handle
// 参数：name - 输出handle名称
// 返回：*Tensor - 输出 handle
func (p *Predictor) GetOutputHandle(name string) *Tensor

// 执行预测
// 参数：无
// 返回：None
func (p *Predictor) Run()

// 释放中间Tensor
// 参数：None
// 返回：None
func (p *Predictor) ClearIntermediateTensor()

// 释放内存池中的所有临时 Tensor
// 参数：None
// 返回：None
func (p *Predictor) TryShrinkMemory()
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
    output := predictor.GetOutputTensors(predictor.GetOutputNames()[0])

    inputData := make([]float32, 1 * 3 * 224 * 224)
    for i := 0; i < 1 * 3 * 224 * 224; i++ {
      inputData[i] = 1.0
    }
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
