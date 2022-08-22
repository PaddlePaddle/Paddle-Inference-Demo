# 枚举类型

## DataType

DataType 为模型中 Tensor 的数据精度，默认值为 `FLOAT32`。枚举变量与 API 定义如下：

```c++
// DataType 枚举类型定义
enum DataType {
  FLOAT32,
  INT64,
  INT32,
  UINT8,
  INT8,
  FLOAT16,
};

// 获取各个 DataType 对应的字节数
// 参数：dtype - DataType 枚举
// 输出：int - 字节数
int GetNumBytesOfDataType(DataType dtype)
```

代码示例：


```c++
// 创建 FLOAT32 类型 DataType
auto data_type = paddle_infer::DataType::FLOAT32;

// 输出 data_type 的字节数 - 4
std::cout << paddle_infer::GetNumBytesOfDataType(data_type) << std::endl;
```

## PrecisionType

PrecisionType 设置模型的运行精度，默认值为 `kFloat32(float32)`。枚举变量定义如下：

```c++
// PrecisionType 枚举类型定义
enum class PrecisionType {
  kFloat32 = 0,  ///< fp32
  kInt8,         ///< int8
  kHalf,         ///< fp16
};
```

代码示例：

```c++
// 创建 Config 对象
paddle_infer::Config config("./mobilenet.pdmodel", "./mobilenet.pdiparams");

// 启用 GPU 进行预测
config.EnableUseGpu(100, 0);

// 启用 TensorRT 进行预测加速 - FP16
config.EnableTensorRtEngine(1 << 28, 1, 3, 
                            paddle_infer::PrecisionType::kHalf, false, false);
```


## PlaceType

PlaceType 为目标设备硬件类型，用户可以根据应用场景选择硬件平台类型。枚举变量定义如下：

```c++
// PlaceType 枚举类型定义
enum class PlaceType { kUNK = -1, kCPU, kGPU, kXPU, kNPU, kIPU, kCUSTOM };
```

代码示例：

```c++
// 创建 Config 对象
paddle_infer::Config config("./mobilenet.pdmodel", "./mobilenet.pdiparams");

// 启用 GPU 预测
config.EnableUseGpu(100, 0);

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

// 获取 Output Tensor 的 PlaceType 和 数据指针
paddle_infer::PlaceType place;
int size = 0;
auto* out_data = output_tensor->data<float>(&place, &size);

// 输出 Place 结果 - true
std::cout << (place == paddle_infer::PlaceType::kGPU) << std::endl;
std::cout << size / sizeof(float) << std::endl;
```
