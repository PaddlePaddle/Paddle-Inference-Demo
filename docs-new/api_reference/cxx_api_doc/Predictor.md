# Predictor 类

Paddle Inference 的预测器，由 `CreatePredictor` 根据 `Config` 进行创建。用户可以根据Predictor提供的接口设置输入数据、执行模型预测、获取输出等。

**注意事项：**
一个 `Config` 对象只能用于调用一次 `CreatePredictor` 生成一个 `Predictor`，需要通过 `CreatePredictor` 创建多个 `Predictor` 时请分别创建 `Config` 对象。

## 获取输入输出

API 定义如下：

```c++
// 获取所有输入 Tensor 的名称
// 参数：None
// 返回：std::vector<std::string> - 所有输入 Tensor 的名称
std::vector<std::string> GetInputNames();

// 根据名称获取输入 Tensor 的句柄
// 参数：name - Tensor 的名称
// 返回：std::unique_ptr<Tensor> - 指向 Tensor 的指针
std::unique_ptr<Tensor> GetInputHandle(const std::string& name);

// 获取所有输出 Tensor 的名称
// 参数：None
// 返回：std::vector<std::string> - 所有输出 Tensor 的名称
std::vector<std::string> GetOutputNames();

// 根据名称获取输出 Tensor 的句柄
// 参数：name - Tensor 的名称
// 返回：std::unique_ptr<Tensor> - 指向 Tensor 的指针
std::unique_ptr<Tensor> GetOutputHandle(const std::string& name);
```

代码示例：

```c++
// 构造 Config 对象
paddle_infer::Config config("./resnet.pdmodel", "./resnet.pdiparams");

// 创建 Predictor
auto predictor = paddle_infer::CreatePredictor(config);

// 准备输入数据
int input_num = shape_production(INPUT_SHAPE);
std::vector<float> input_data(input_num, 1);

// 准备输入 Tensor
auto input_names = predictor->GetInputNames();
auto input_tensor = predictor->GetInputHandle(input_names[0]);
input_tensor->Reshape({1, 3, 224, 224});
input_tensor->CopyFromCpu(input_data.data());

// 执行预测
predictor->Run();

// 获取 Output Tensor
auto output_names = predictor->GetOutputNames();
auto output_tensor = predictor->GetOutputHandle(output_names[0]);
```

## 运行和生成

API 定义如下：

```c++
// 执行模型预测，需要在设置输入数据后调用
// 参数：None
// 返回：None
bool Run();

// 根据该 Predictor，克隆一个新的 Predictor，两个 Predictor 之间共享权重
// 参数：None
// 返回：std::unique_ptr<Predictor> - 新的 Predictor
std::unique_ptr<Predictor> Clone();

// 释放中间Tensor
// 参数：None
// 返回：None
void ClearIntermediateTensor();

// 释放内存池中的所有临时 Tensor
// 参数：None
// 返回：uint64_t - 释放的内存字节数
uint64_t TryShrinkMemory();
```

代码示例：

```c++
// 创建 Predictor
auto predictor = paddle_infer::CreatePredictor(config);

// 准备输入数据
int input_num = shape_production(INPUT_SHAPE);
std::vector<float> input_data(input_num, 1);

// 准备输入 Tensor
auto input_names = predictor->GetInputNames();
auto input_tensor = predictor->GetInputHandle(input_names[0]);
input_tensor->Reshape({1, 3, 224, 224});
input_tensor->CopyFromCpu(input_data.data());

// 执行预测
predictor->Run();

// 获取 Output Tensor
auto output_names = predictor->GetOutputNames();
auto output_tensor = predictor->GetOutputHandle(output_names[0]);
std::vector<int> output_shape = output_tensor->shape();
int out_num = std::accumulate(output_shape.begin(), output_shape.end(), 
                              1, std::multiplies<int>());
// 获取 Output 数据
std::vector<float> out_data;
out_data.resize(out_num);
output_tensor->CopyToCpu(out_data.data());

// 释放中间Tensor
predictor->ClearIntermediateTensor();

// 释放内存池中的所有临时 Tensor
predictor->TryShrinkMemory();
```