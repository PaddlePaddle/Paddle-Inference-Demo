# 枚举类型

## DataType

DataType为模型中Tensor的数据精度，默认值为 `FLOAT32`。枚举变量与 API 定义如下：

```c
// DataType 枚举类型定义
enum PD_DataType { PD_FLOAT32, PD_INT32, PD_INT64, PD_UINT8, PD_UNKDTYPE };
```

代码示例：

```c
// 创建 FLOAT32 类型 DataType
PD_DataType data_type = PD_FLOAT32;

// 创建输入 Tensor
PD_Tensor* input_tensor = PD_NewPaddleTensor();

// 设置输入 Tensor 的数据类型
PD_SetPaddleTensorDType(input_tensor, PD_FLOAT32);

// 获取 Tensor 的数据类型
printf("Tensor Dtype: %d\n", PD_GetPaddleTensorDType(input_tensor));
```

## Precision

Precision 设置模型的运行精度，默认值为 `kFloat32(float32)`。枚举变量定义如下：

```c
// PrecisionType 枚举类型定义
enum Precision { kFloat32 = 0, kInt8, kHalf };
```

代码示例：

```c
// 创建 AnalysisConfig 对象
PD_AnalysisConfig* config = PD_NewAnalysisConfig();

// 启用 GPU 进行预测 - 初始化 GPU 显存 100M, Deivce_ID 为 0
PD_EnableUseGpu(config, 100, 0);

// 启用 TensorRT 进行预测加速 - FP32
PD_EnableTensorRtEngine(config, 1 << 20, 1, 3, kFloat32, false, false);

// 启用 TensorRT 进行预测加速 - FP16
PD_EnableTensorRtEngine(config, 1 << 20, 1, 3, kHalf, false, false);

// 启用 TensorRT 进行预测加速 - Int8
PD_EnableTensorRtEngine(config, 1 << 20, 1, 3, kInt8, false, false);
```
