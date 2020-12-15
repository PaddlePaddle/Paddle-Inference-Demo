#  PaddleTensor 方法

PaddleTensor 是 Paddle Inference 的数据组织形式，用于对底层数据进行封装并提供接口对数据进行操作，包括设置 Shape、数据、LoD 信息等。

## 创建 PaddleTensor 对象

```c
// 创建 PaddleTensor 对象
// 参数：None
// 返回：PD_Tensor* - PaddleTensor 对象指针
PD_Tensor* PD_NewPaddleTensor();

// 删除 PaddleTensor 对象
// 参数：tensor - PaddleTensor 对象指针
// 返回：None
void PD_DeletePaddleTensor(PD_Tensor* tensor);
```

代码示例:

```c
// 创建 PaddleTensor 对象
PD_Tensor* input = PD_NewPaddleTensor();

// 删除 PaddleTensor 对象
PD_DeletePaddleTensor(input);
```

## 输入输出 PaddleTensor

```c
// 设置 PaddleTensor 名称
// 参数：tensor - PaddleTensor 对象指针
//      name - PaddleTensor 名称
// 返回：None
void PD_SetPaddleTensorName(PD_Tensor* tensor, char* name);

// 设置 PaddleTensor 数据类型
// 参数：tensor - PaddleTensor 对象指针
//      dtype - PaddleTensor 数据类型，DataType 类型
// 返回：None
void PD_SetPaddleTensorDType(PD_Tensor* tensor, PD_DataType dtype);

// 设置 PaddleTensor 数据
// 参数：tensor - PaddleTensor 对象指针
//      buf - PaddleTensor 数据，PaddleBuf 类型指针
// 返回：None
void PD_SetPaddleTensorData(PD_Tensor* tensor, PD_PaddleBuf* buf);

// 设置 PaddleTensor 的维度信息
// 参数：tensor - PaddleTensor 对象指针
//      shape - 包含维度信息的int数组指针
//      size - 包含维度信息的int数组长度
// 返回：None
void PD_SetPaddleTensorShape(PD_Tensor* tensor, int* shape, int size);

// 获取 PaddleTensor 名称
// 参数：tensor - PaddleTensor 对象指针
// 返回：const char * - PaddleTensor 名称
const char* PD_GetPaddleTensorName(const PD_Tensor* tensor);

// 获取 PaddleTensor 数据类型
// 参数：tensor - PaddleTensor 对象指针
// 返回：PD_DataType- PaddleTensor 数据类型
PD_DataType PD_GetPaddleTensorDType(const PD_Tensor* tensor);

// 获取 PaddleTensor 数据
// 参数：tensor - PaddleTensor 对象指针
// 返回：PD_PaddleBuf * - PaddleTensor 数据
PD_PaddleBuf* PD_GetPaddleTensorData(const PD_Tensor* tensor);

// 获取 PaddleTensor 唯独信息
// 参数：tensor - PaddleTensor 对象指针
//      size - (可修改参数) 返回包含 PaddleTensor 维度信息的int数组长度
// 返回：const int* - 包含 PaddleTensor 维度信息的int数组指针Tensor 
const int* PD_GetPaddleTensorShape(const PD_Tensor* tensor, int* size);
```

代码示例：

```c
// 创建 AnalysisConfig 对象
PD_AnalysisConfig* config = PD_NewAnalysisConfig();

// 设置预测模型路径，这里为非 Combined 模型
const char* model_dir  = "./mobilenet_v1";
PD_SetModel(config, model_dir, NULL);

// 创建输入 PaddleTensor
PD_Tensor* input_tensor = PD_NewPaddleTensor();
// 创建输入 Buffer
PD_PaddleBuf* input_buffer = PD_NewPaddleBuf();
printf("PaddleBuf empty: %s\n", PD_PaddleBufEmpty(input_buffer) ? "True" : "False");
int batch = 1;
int channel = 3;
int height = 224;
int width = 224;
int input_shape[4] = {batch, channel, height, width};
int input_size = batch * channel * height * width;
int shape_size = 4;
float* input_data  = malloc(sizeof(float) * input_size);
int i = 0;
for (i = 0; i < input_size ; i++){ 
  input_data[i] = 1.0f; 
}
PD_PaddleBufReset(input_buffer, (void*)(input_data), sizeof(float) * input_size);

// 设置输入 PaddleTensor 信息
char name[6] = {'i', 'm', 'a', 'g', 'e', '\0'};
PD_SetPaddleTensorName(input_tensor, name);
PD_SetPaddleTensorDType(input_tensor, PD_FLOAT32);
PD_SetPaddleTensorShape(input_tensor, input_shape, shape_size);
PD_SetPaddleTensorData(input_tensor, input_buffer);

// 设置输出 PaddleTensor 和 数量
PD_Tensor* output_tensor = PD_NewPaddleTensor();
int output_size;

// 执行预测
PD_PredictorRun(config, input_tensor, 1, &output_tensor, &output_size, 1);

// 获取预测输出 PaddleTensor 信息
printf("Output PaddleTensor Size: %d\n", output_size);
printf("Output PaddleTensor Name: %s\n", PD_GetPaddleTensorName(output_tensor));
printf("Output PaddleTensor Dtype: %d\n", PD_GetPaddleTensorDType(output_tensor));

// 获取预测输出 PaddleTensor 数据
PD_PaddleBuf* output_buffer = PD_GetPaddleTensorData(output_tensor);
float* result = (float*)(PD_PaddleBufData(output_buffer));
int result_length = PD_PaddleBufLength(output_buffer) / sizeof(float);
printf("Output Data Length: %d\n", result_length);

// 删除输入 PaddleTensor 和 Buffer
PD_DeletePaddleTensor(input_tensor);
PD_DeletePaddleBuf(input_buffer);

// 删除 Config
PD_DeleteAnalysisConfig(config);
```