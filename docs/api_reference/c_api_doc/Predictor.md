# Predictor 方法

Paddle Inference 的预测器，由 `PD_NewPredictor` 根据 `AnalysisConfig` 进行创建。用户可以根据 Predictor 提供的接口设置输入数据、执行模型预测、获取输出等。

## 创建 Predictor

API定义如下：

```c
// 根据 Config 构建预测执行对象 Predictor
// 参数: config - 用于构建 Predictor 的配置信息
// 返回: PD_Predictor* - 预测对象指针
PD_Predictor* PD_NewPredictor(const PD_AnalysisConfig* config);

// 删除 Predictor 对象
// predictor - Predictor 对象指针
// 返回：None
void PD_DeletePredictor(PD_Predictor* predictor);
```

代码示例:

```c
// 创建 AnalysisConfig 对象
PD_AnalysisConfig* config = PD_NewAnalysisConfig();

// 设置预测模型路径，这里为非 Combined 模型
const char* model_dir  = "./mobilenet_v1";
PD_SetModel(config, model_dir, NULL);

// 根据 Config 创建 Predictor
PD_Predictor* predictor = PD_NewPredictor(config);
```

## 获取输入输出

API 定义如下：

```c
// 获取模型输入 Tensor 的数量
// 参数：predictor - PD_Predictor 对象指针
// 返回：int - 模型输入 Tensor 的数量
int PD_GetInputNum(const PD_Predictor*);

// 获取模型输出 Tensor 的数量
// 参数：predictor - PD_Predictor 对象指针
// 返回：int - 模型输出 Tensor 的数量
int PD_GetOutputNum(const PD_Predictor*);

// 获取输入 Tensor 名称
// 参数：predictor - PD_Predictor 对象指针
//      int - 输入 Tensor 的index
// 返回：const char* - 输入 Tensor 名称
const char* PD_GetInputName(const PD_Predictor*, int);

// 获取输出 Tensor 名称
// 参数：predictor - PD_Predictor 对象指针
//      int - 输出 Tensor 的index
// 返回：const char* - 输出 Tensor 名称
const char* PD_GetOutputName(const PD_Predictor*, int);
```

代码示例：

```c
// 创建 AnalysisConfig 对象
PD_AnalysisConfig* config = PD_NewAnalysisConfig();

// 设置预测模型路径，这里为非 Combined 模型
const char* model_dir  = "./mobilenet_v1";
PD_SetModel(config, model_dir, NULL);

// 根据 Config 创建 Predictor
PD_Predictor* predictor = PD_NewPredictor(config);

// 获取输入 Tensor 的数量
int input_num = PD_GetInputNum(predictor);
printf("Input tensor number is: %d\n", input_num);

// 获取第 0 个输入 Tensor的名称
const char * input_name = PD_GetInputName(predictor, 0);
printf("Input tensor name is: %s\n", input_name);

// 获取输出 Tensor 的数量
int output_num = PD_GetOutputNum(predictor);
printf("Output tensor number is: %d\n", output_num);

// 获取第 0 个输出 Tensor的名称
const char * output_name = PD_GetOutputName(predictor, 0);
printf("Output tensor name is: %s\n", output_name);
```

## 执行预测

API 定义如下：

```c
// 执行模型预测，需要在设置输入数据后调用
// 参数：config - 用于构建 Predictor 的配置信息
//      inputs - 输入 Tensor 的数组指针
//      in_size - 输入 Tensor 的数组中包含的输入 Tensor 的数量
//      output_data - (可修改参数) 返回输出 Tensor 的数组指针
//      out_size - (可修改参数) 返回输出 Tensor 的数组中包含的输出 Tensor 的数量
//      batch_size - 输入的 batch_size
// 返回：bool - 执行预测是否成功
bool PD_PredictorRun(const PD_AnalysisConfig* config,
                     PD_Tensor* inputs, int in_size,
                     PD_Tensor** output_data,
                     int* out_size, int batch_size);
```

代码示例：

```c
// 创建 AnalysisConfig 对象
PD_AnalysisConfig* config = PD_NewAnalysisConfig();

// 设置预测模型路径，这里为非 Combined 模型
const char* model_dir  = "./mobilenet_v1";
PD_SetModel(config, model_dir, NULL);

// 创建输入 Tensor
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

// 设置输入 Tensor 信息
char name[6] = {'i', 'm', 'a', 'g', 'e', '\0'};
PD_SetPaddleTensorName(input_tensor, name);
PD_SetPaddleTensorDType(input_tensor, PD_FLOAT32);
PD_SetPaddleTensorShape(input_tensor, input_shape, shape_size);
PD_SetPaddleTensorData(input_tensor, input_buffer);

// 设置输出 Tensor 和 数量
PD_Tensor* output_tensor = PD_NewPaddleTensor();
int output_size;

// 执行预测
PD_PredictorRun(config, input_tensor, 1, &output_tensor, &output_size, 1);

// 获取预测输出 Tensor 信息
printf("Output Tensor Size: %d\n", output_size);
printf("Output Tensor Name: %s\n", PD_GetPaddleTensorName(output_tensor));
printf("Output Tensor Dtype: %d\n", PD_GetPaddleTensorDType(output_tensor));

// 获取预测输出 Tensor 数据
PD_PaddleBuf* output_buffer = PD_GetPaddleTensorData(output_tensor);
float* result = (float*)(PD_PaddleBufData(output_buffer));
int result_length = PD_PaddleBufLength(output_buffer) / sizeof(float);
printf("Output Data Length: %d\n", result_length);

// 删除输入 Tensor 和 Buffer
PD_DeletePaddleTensor(input_tensor);
PD_DeletePaddleBuf(input_buffer);

// 删除 Config
PD_DeleteAnalysisConfig(config);
```