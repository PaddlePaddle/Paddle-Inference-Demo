#  Tensor 类

Tensor 是 Paddle Inference 的数据组织形式，用于对底层数据进行封装并提供接口对数据进行操作，包括设置 Shape、数据、LoD 信息等。在 Paddle Inference 中存在两种 Tensor，分别是 paddle_infer::Tensor 和 paddle::Tensor， 其中 paddle::Tensor 为训推一体共用，对应于 Python 端的 Paddle.Tensor。在 Paddle2.5 及之后的版本中，我们推荐使用 paddle::Tensor。

## 一、paddle_infer::Tensor

**注意：** 
应使用 `Predictor` 的 `GetInputHandle` 和 `GetOuputHandle` 接口获取输入输出 `Tensor`。

paddle_infer::Tensor 类的API定义如下：

```c++
// 设置 Tensor 的维度信息
// 参数：shape - 维度信息
// 返回：None
void Reshape(const std::vector<int>& shape);

// 从 CPU 获取数据，设置到 Tensor 内部
// 参数：data - CPU 数据指针
// 返回：None
template <typename T>
void CopyFromCpu(const T* data);

// 从 Tensor 中获取数据到 CPU，该接口内含同步等待 GPU 运行结束，当 Predictor 
// 运行在 GPU 硬件时，在 CPU 线程下对该 API 调用进行计时是不准确的
//
// 参数：data - CPU 数据指针
// 返回：None
template <typename T>
void CopyToCpu(T* data);

// 使用用户的数据指针创建输入/输出 Tensor
// 创建输入 Tensor 时，用户保证输入指针数据预测过程中有效
// 创建输出 Tensor 时，用户保证输出指针的数据长度大于等于模型的输出数据大小
// 参数：data - CPU/GPU 数据指针
// 参数：shape - 数据 shape
// 参数：place - 数据的存放位置
// 参数：layout - 数据格式，默认为 NCHW，当前仅支持 NCHW
// 返回：None
template <typename T>
void ShareExternalData(const T* data, const std::vector<int>& shape,
                       PlaceType place, DataLayout layout = DataLayout::kNCHW);

// 获取 Tensor 底层数据指针，用于设置 Tensor 输入数据
// 在调用这个 API 之前需要先对输入 Tensor 进行 Reshape
// 参数：place - 获取 Tensor 的 PlaceType
// 返回：数据指针
template <typename T>
T* mutable_data(PlaceType place);

// 获取 Tensor 底层数据的常量指针，用于读取 Tensor 输出数据
// 参数：place - 获取 Tensor 的 PlaceType
//      size - 获取 Tensor 的 size
// 返回：数据指针
template <typename T>
T* data(PlaceType* place, int* size) const;

// 设置 Tensor 的 LoD 信息
// 参数：x - Tensor 的 LoD 信息
// 返回：None
void SetLoD(const std::vector<std::vector<size_t>>& x);

// 获取 Tensor 的 LoD 信息
// 参数：None
// 返回：std::vector<std::vector<size_t>> - Tensor 的 LoD 信息
std::vector<std::vector<size_t>> lod() const;

// 获取 Tensor 的 DataType
// 参数：None
// 返回：DataType - Tensor 的 DataType
DataType type() const;

// 获取 Tensor 的维度信息
// 参数：None
// 返回：std::vector<int> - Tensor 的维度信息
std::vector<int> shape() const;

// 获取 Tensor 的 Name
// 参数：None
// 返回：std::string& - Tensor 的 Name
const std::string& name() const;
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

// 获取输入 Tensor
auto input_names = predictor->GetInputNames();
auto input_tensor = predictor->GetInputHandle(input_names[0]);

// 设置输入 Tensor 的维度信息
input_tensor->Reshape(INPUT_SHAPE);
// 获取输入 Tensor 的 Name
auto name = input_tensor->name();

//  方式1: 通过 mutable_data 设置输入数据
std::copy_n(input_data.begin(), input_data.size(),
            input_tensor->mutable_data<float>(PaddlePlace::kCPU));

//  方式2: 通过 CopyFromCpu 设置输入数据
input_tensor->CopyFromCpu(input_data.data());

//  方式3: 通过 ShareExternalData 设置输入数据
input_tensor->ShareExternalData<float>(input, INPUT_SHAPE, PlaceType::kCPU);

// 执行预测
predictor->Run();

// 获取 Output Tensor
auto output_names = predictor->GetOutputNames();
auto output_tensor = predictor->GetOutputHandle(output_names[0]);

// 获取 Output Tensor 的维度信息
std::vector<int> output_shape = output_tensor->shape();

// 方式1: 通过 data 获取 Output Tensor 的数据
paddle_infer::PlaceType place;
int size = 0;
auto* out_data = output_tensor->data<float>(&place, &size);

// 方式2: 通过 CopyToCpu 获取 Output Tensor 的数据
std::vector<float> output_data;
output_data.resize(output_size);
output_tensor->CopyToCpu(output_data.data());
```

## 二、paddle::Tensor

[参考官网链接](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/custom_op/new_cpp_op_cn.html#tensor-api)。