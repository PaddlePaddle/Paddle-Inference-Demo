#  Tensor 方法

Tensor 是 Paddle Inference 的数据组织形式，用于对底层数据进行封装并提供接口对数据进行操作，包括设置 Shape、数据、LoD 信息等。

**注意：** 应使用 `PD_PredictorGetInputHandle` 和 `PD_PredictorGetOutputHandle` 接口获取输入输出 `Tensor`。

Tensor 相关的API定义如下：

```c
// 设置 Tensor 的维度信息
// 参数：pd_tensor - Tensor 对象指针
//      shape_size - 维度信息的长度
//      shape - 维度信息指针
// 返回：None
void PD_TensorReshape(PD_Tensor* pd_tensor, size_t shape_size, int32_t* shape);

// 从 CPU 获取 float / int64_t / int32_t / uint8_t / int8_t 类型数据，拷贝到 Tensor 内部
// 参数：pd_tensor - Tensor 对象指针
//      data - CPU 数据指针
// 返回：None
void PD_TensorCopyFromCpuFloat(PD_Tensor* pd_tensor, const float* data);
void PD_TensorCopyFromCpuInt64(PD_Tensor* pd_tensor, const int64_t* data);
void PD_TensorCopyFromCpuInt32(PD_Tensor* pd_tensor, const int32_t* data);
void PD_TensorCopyFromCpuUint8(PD_Tensor* pd_tensor, const uint8_t* data);
void PD_TensorCopyFromCpuInt8(PD_Tensor* pd_tensor, const int8_t* data);

// 从 Tensor 中获取 float / int64_t / int32_t / uint8_t / int8_t 类型数据，拷贝到 CPU
// 参数：pd_tensor - Tensor 对象指针
//      data - CPU 数据指针
// 返回：None
void PD_TensorCopyToCpuFloat(PD_Tensor* pd_tensor, const float* data);
void PD_TensorCopyToCpuInt64(PD_Tensor* pd_tensor, const int64_t* data);
void PD_TensorCopyToCpuInt32(PD_Tensor* pd_tensor, const int32_t* data);
void PD_TensorCopyToCpuUint8(PD_Tensor* pd_tensor, const uint8_t* data);
void PD_TensorCopyToCpuInt8(PD_Tensor* pd_tensor, const int8_t* data);

// 获取 Tensor 底层数据指针，用于设置 Tensor 输入数据
// 在调用这个 API 之前需要先对输入 Tensor 进行 Reshape
// 参数：pd_tensor - Tensor 对象指针
//      place - 获取 Tensor 的 PlaceType
// 返回：数据指针
float* PD_TensorMutableDataFloat(PD_Tensor* pd_tensor, PD_PlaceType place);
int64_t* PD_TensorMutableDataInt64(PD_Tensor* pd_tensor, PD_PlaceType place);
int32_t* PD_TensorMutableDataInt32(PD_Tensor* pd_tensor, PD_PlaceType place);
uint8_t* PD_TensorMutableDataUint8(PD_Tensor* pd_tensor, PD_PlaceType place);
int8_t* PD_TensorMutableDataInt8(PD_Tensor* pd_tensor, PD_PlaceType place);

// 获取 Tensor 底层数据的常量指针，用于读取 Tensor 输出数据
// 参数：pd_tensor - Tensor 对象指针
//      place - 获取 Tensor 的 PlaceType
//      size - 获取 Tensor 的 size
// 返回：数据指针
float* PD_TensorDataFloat(PD_Tensor* pd_tensor, PD_PlaceType* place, int32_t* size);
int64_t* PD_TensorDataInt64(PD_Tensor* pd_tensor, PD_PlaceType* place, int32_t* size);
int32_t* PD_TensorDataInt32(PD_Tensor* pd_tensor, PD_PlaceType* place, int32_t* size);
uint8_t* PD_TensorDataUint8(PD_Tensor* pd_tensor, PD_PlaceType* place, int32_t* size);
int8_t* PD_TensorDataInt8(PD_Tensor* pd_tensor, PD_PlaceType* place, int32_t* size);

// 设置 Tensor 的 LoD 信息
// 参数：pd_tensor - Tensor 对象指针
//      lod - Tensor 的 LoD 信息
// 返回：None
void PD_TensorSetLod(PD_Tensor* pd_tensor, PD_TwoDimArraySize* lod);

// 获取 Tensor 的 LoD 信息
// 参数：pd_tensor - Tensor 对象指针
// 返回：PD_TwoDimArraySize* - Tensor 的 LoD 信息。
//      该LoD信息对象需要通过 ’PD_TwoDimArraySizeDestroy‘ 进行回收。
PD_TwoDimArraySize* PD_TensorGetLod(PD_Tensor* pd_tensor);

// 获取 Tensor 的 DataType
// 参数：pd_tensor - Tensor 对象指针
// 返回：PD_DataType - Tensor 的 DataType
PD_DataType PD_TensorGetDataType(PD_Tensor* pd_tensor);

// 获取 Tensor 的维度信息
// 参数：pd_tensor - Tensor 对象指针
// 返回：PD_OneDimArrayInt32* - Tensor 的维度信息
//      该返回值需要通过 ’PD_OneDimArrayInt32Destroy‘ 进行回收。
PD_OneDimArrayInt32* PD_TensorGetShape(PD_Tensor* pd_tensor);

// 获取 Tensor 的 Name
// 参数：pd_tensor - Tensor 对象指针
// 返回：const char* - Tensor 的 Name
const char* PD_TensorGetName(PD_Tensor* pd_tensor);

// 销毁 Tensor 对象
// 参数：pd_tensor - Tensor 对象指针
// 返回：None
void PD_TensorDestroy(__pd_take PD_Tensor* pd_tensor);
```

代码示例：

```c
// 创建 Config 对象
PD_Config* config = PD_ConfigCreate();

// 设置预测模型路径，这里为 Combined 模型
const char* model_path  = "./model/inference.pdmodel";  
const char* params_path = "./model/inference.pdiparams";
PD_ConfigSetModel(config, model_path, params_path);

// 根据 Config 创建 Predictor, 并销毁 Config 对象
PD_Predictor* predictor = PD_PredictorCreate(config);

// 准备输入数据
float input_shape[4] = {1, 3, 244, 244};
float input_data = (float*)calloc(1 * 3 * 224 * 224, sizeof(float));

// 获取输入 Tensor
PD_OneDimArrayCstr* input_names = PD_PredictorGetInputNames(predictor);
PD_Tensor* input_tensor = PD_PredictorGetInputHandle(predictor, input_names->data[0]);

// 设置输入 Tensor 的维度信息
PD_TensorReshape(input_tensor, 4, input_shape);

// 获取输入 Tensor 的 Name
const char* name = PD_TensorGetName(PD_Tensor* pd_tensor);

//  方式1: 通过 mutable_data 设置输入数据
float* data_ptr = PD_TensorMutableDataFloat(pd_tensor, PD_PLACE_CPU);
memcpy(data_ptr, input_data, 1 * 3 * 224 * 224 * sizeof(float));

//  方式2: 通过 CopyFromCpu 设置输入数据
PD_TensorCopyFromCpuFloat(input_tensor, input_data);

// 执行预测
PD_PredictorRun(pd_predictor);

// 获取预测输出 Tensor
PD_OneDimArrayCstr* output_names = PD_PredictorGetOutputNames(predictor);
PD_Tensor* output_tensor = PD_PredictorGetOutputHandle(predictor, output_names->data[0]);

// 方式1: 通过 data 获取 Output Tensor 的数据
PD_PlaceType place;
int32_t size;
float* out_data_ptr = PD_TensorDataFloat(output_tensor, &place, &size);
float* out_data = (float*)malloc(size * sizeof(float));
memcpy(out_data, out_data_ptr, size * sizeof(float));
free(out_data);

// 方式2: 通过 CopyToCpu 获取 Output Tensor 的数据
PD_OneDimArrayInt32* output_shape = PD_TensorGetShape(output_tensor);
int32_t out_size = 1;
for (size_t i = 0; i < output_shape->size; ++i) {
  out_size = out_size * output_shape->data[i];
}
out_data = (float*)malloc(out_size * sizeof(float));
PD_TensorCopyToCpuFloat(output_tensor, out_data);
free(out_data)

// 销毁相关对象， 回收相关内存
PD_OneDimArrayInt32Destroy(output_shape);
PD_TensorDestroy(output_tensor);
PD_OneDimArrayCstrDestroy(output_names);
PD_TensorDestroy(input_tensor);
PD_OneDimArrayCstrDestroy(input_names);
free(input_data);
PD_PredictorDestroy(predictor);
```
