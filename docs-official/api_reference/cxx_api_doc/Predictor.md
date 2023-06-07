# Predictor 类

Paddle Inference 的预测器，由 `CreatePredictor` 根据 `Config` 进行创建。用户可以根据Predictor提供的接口设置输入数据、执行模型预测、获取输出等。

**注意事项：**
一个 `Config` 对象只能用于调用一次 `CreatePredictor` 生成一个 `Predictor`，需要通过 `CreatePredictor` 创建多个 `Predictor` 时请分别创建 `Config` 对象。

## 获取输入输出

API 定义如下：

```c++
// 获取所有输入 paddle_infer::Tensor 的名称
// 参数：None
// 返回：std::vector<std::string> - 所有输入 paddle_infer::Tensor 的名称
std::vector<std::string> GetInputNames();

// 根据名称获取输入 paddle_infer::Tensor 的句柄
// 参数：name - paddle_infer::Tensor 的名称
// 返回：std::unique_ptr<Tensor> - 指向 paddle_infer::Tensor 的指针
std::unique_ptr<paddle_infer::Tensor> GetInputHandle(const std::string& name);

// 获取所有输出 paddle_infer::Tensor 的名称
// 参数：None
// 返回：std::vector<std::string> - 所有输出 paddle_infer::Tensor 的名称
std::vector<std::string> GetOutputNames();

// 根据名称获取输出 paddle_infer::Tensor 的句柄
// 参数：name - paddle_infer::Tensor 的名称
// 返回：std::unique_ptr<Tensor> - 指向 paddle_infer::Tensor 的指针
std::unique_ptr<paddle_infer::Tensor> GetOutputHandle(const std::string& name);
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

// 准备输入 paddle_infer::Tensor
auto input_names = predictor->GetInputNames();
auto input_tensor = predictor->GetInputHandle(input_names[0]);
input_tensor->Reshape({1, 3, 224, 224});
input_tensor->CopyFromCpu(input_data.data());

// 执行预测
predictor->Run();

// 获取 Output paddle_infer::Tensor
auto output_names = predictor->GetOutputNames();
auto output_tensor = predictor->GetOutputHandle(output_names[0]);
```

## 运行和生成

API 定义如下：

```c++
// 执行模型预测，需要在设置输入数据后调用
// 参数：None
// 返回：bool - 是否执行成功
// 备注：此接口对应于 paddle_infer::Tensor
bool Run();

// 执行模型预测(推荐使用)
// 参数：inputs - 输入数据，对应模型输入的 paddle::Tensor 向量
//      outputs - 输出数据，对应模型输出的 paddle::Tenosr 向量
// 返回：bool - 是否执行成功
// 备注：此接口对应于 paddle::Tensor
bool Run(const std::vector<paddle::Tensor>& inputs, std::vector<paddle::Tensor> * outputs);

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

// 准备输入 paddle_infer::Tensor
auto input_names = predictor->GetInputNames();
auto input_tensor = predictor->GetInputHandle(input_names[0]);
input_tensor->Reshape({1, 3, 224, 224});
input_tensor->CopyFromCpu(input_data.data());

// 执行预测
predictor->Run();

// 获取 Output paddle_infer::Tensor
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

## 获取 OP 中间输出 paddle_infer::Tensor

API 定义如下：

```c++
// 获取中间 op 的输出 paddle::Tensor
// 参数：OutputHookFunc    - hook 函数签名为 void(const std::string&, const std::string&, const paddle::Tensor&)
//                              第一个参数是 op type（name）
//                              第二个参数是输出 paddle::Tensor‘s name
//                              第三个参数是输出 paddle::Tensor
// 返回：None
void RegisterOutputHook(const OutputHookFunc& hookfunc);
```

代码示例：

示例一：
该示例输出所有 tensor 的均值和方差。用户可根据需要单独输出某一特定 op 的输出 tensor 的信息。

（下面是针对跑 fp32 的模型给出的示例。跑混合精度的话，需要做些修改，具体见注释部分）
```cpp
// 使用 paddle::Tensor 需要包含这个头文件
#include "paddle/extension.h"

void get_output_tensor(const std::string &op_type,
                       const std::string &tensor_name,
                       const paddle::Tensor &tensor) {
  if(tensor.dtype() != paddle::DataType::FLOAT32) return;
  auto cpu_tensor = tensor.copy_to(paddle::CPUPlace{}, true);
  // using TYPE = phi::dtype::float16;
  using TYPE = float;

  std::stringstream ss;

  // op type and tensor name
  ss << std::left << std::setw(20) << op_type << std::setw(40) << tensor_name;

  // tensor shape
  std::string shape_str;
  shape_str += "[" + std::to_string(cpu_tensor.shape()[0]);
  for (size_t i = 1; i < cpu_tensor.shape().size(); i++) {
    shape_str += "," + std::to_string(cpu_tensor.shape()[i]);
  }
  shape_str += "]";
  ss << std::setw(20) << shape_str;

  // tensor data mean and variance
  TYPE sum{0};
  for (size_t i = 0; i < cpu_tensor.numel(); i++) {
    sum += cpu_tensor.data<TYPE>()[i];
  }
  TYPE mean = sum / TYPE(cpu_tensor.numel());
  TYPE accum{0};
  for (size_t i = 0; i < cpu_tensor.numel(); i++) {
    accum += (cpu_tensor.data<TYPE>()[i] - mean) *
             (cpu_tensor.data<TYPE>()[i] - mean);
  }
  TYPE variance = accum / TYPE(cpu_tensor.numel());
  ss << std::setw(20) << mean << std::setw(20) << variance;

  std::cout << ss.str() << std::endl;
}

// 通过该接口注册的 hook 函数，在每个 op run 完都会被执行一次
predictor->RegisterOutputHook(get_output_tensor);
```

输出结果（`op type`、`output tensor name`、`tensor shape`、`mean of tensor`、`variance of tensor`）：

![image](https://user-images.githubusercontent.com/23653004/195584773-42bc2e95-87b0-40b4-9b61-48adb5fa142a.png)

示例二：
该示例输出每个 op run 前后当前 device 上的显存占用信息。
```cpp
void get_current_memory(const std::string &op_type,
                        const std::string &tensor_name, const paddle::Tensor &tensor) {
  // parameters tensor_name and tensor are not used
  std::stringstream ss;

  int device_id = 0;
  if (auto p = getenv("CUDA_VISIBLE_DEVICES")) {
    device_id = atoi(p);
  }
  size_t avail;
  size_t total;
  cudaMemGetInfo(&avail, &total);
  ss << std::left << std::setw(20) << op_type << std::setw(5) << device_id
     << std::setw(10) << (total - avail) / 1024.0 / 1024.0 << std::setw(5)
     << 1.0 * (total - avail) / total;
  LOG(INFO) << ss.str();
}

// 通过该接口注册的 hook 函数，在每个 op run 完都会被执行一次
predictor->RegisterOutputHook(get_current_memory);
```

输出结果（`op type`、`device id`、`memory usage(MiB)`、`memory usage(%)`）：

![image](https://user-images.githubusercontent.com/23653004/196133166-7705c00b-2d0a-499d-bfae-39ecb5b1e9e4.png)
