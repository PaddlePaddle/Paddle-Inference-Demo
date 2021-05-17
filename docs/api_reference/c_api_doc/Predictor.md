# Predictor 方法

Paddle Inference 的预测器，由 `PD_PredictorCreate` 根据 `Config` 进行创建。用户可以根据 Predictor 提供的接口设置输入数据、执行模型预测、获取输出等。

## 创建 Predictor

API定义如下：

```c
// 根据 Config 构建预测执行对象 Predictor, 并销毁传入的Config对象
// 参数：pd_config - 用于构建 Predictor 的配置信息
// 返回：PD_Predictor* - 预测对象指针
PD_Predictor* PD_PredictorCreate(PD_Config* pd_config);

// 根据 已有的 Predictor 对象克隆一个新的 Predictor 对象
// 参数：pd_predictor - 用于克隆新对象的 Predictor 指针
// 返回：PD_Predictor* - 一个新的 Predictor 对象
PD_Predictor* PD_PredictorClone(PD_Predictor* pd_predictor);

// 销毁 Predictor 对象
// 参数：pd_predictor - Predictor 对象指针
// 返回：None
void PD_PredictorDestroy(PD_Predictor* pd_predictor);
```

代码示例:

```c
// 创建 Config 对象
PD_Config* config = PD_ConfigCreate();

// 设置预测模型路径，这里为 Combined 模型
const char* model_path  = "./model/inference.pdmodel";  
const char* params_path = "./model/inference.pdiparams";
PD_ConfigSetModel(config, model_path, params_path);

// 根据 Config 创建 Predictor, 并销毁 Config 对象
PD_Predictor* predictor = PD_PredictorCreate(config);

// 根据已有的 Predictor 克隆出一个新的 Predictor 对象
PD_Predictor* new_predictor = PD_PredictorClone(predictor);

// 销毁 Predictor 对象
PD_PredictorDestroy(new_predictor);
PD_PredictorDestroy(predictor);
```

## 获取输入输出

API 定义如下：

```c
// 获取输入 Tensor 名称
// 参数：pd_predictor - Predictor 对象指针
// 返回：PD_OneDimArrayCstr* - 由输入 Tensor 名称构成的一维字符串数组。
//      该一维字符串数组需要显式调用`PD_OneDimArrayCstrDestroy`来销毁。
PD_OneDimArrayCstr* PD_PredictorGetInputNames(PD_Predictor* pd_predictor);

// 获取输入 Tensor 数量
// 参数：pd_predictor - Predictor 对象指针
// 返回：size_t - Predictor 的输入 tensor 数量。
size_t PD_PredictorGetInputNum(PD_Predictor* pd_predictor);

// 根据名称获取输入 Tensor 的句柄
// 参数：pd_predictor - Predictor 对象指针
//      name - Tensor 的名称
// 返回：PD_Tensor* - 指向 Tensor 的指针。
//      该 Tensor 需要显式调用`PD_TensorDestroy`来销毁。
PD_Tensor* PD_PredictorGetInputHandle(PD_Predictor* pd_predictor, const char* name);

// 获取输出 Tensor 名称
// 参数：pd_predictor - Predictor 对象指针
// 返回：PD_OneDimArrayCstr* - 由输出 Tensor 名称构成的一维字符串数组。
//      该一维字符串数组需要显式调用`PD_OneDimArrayCstrDestroy`来销毁。
PD_OneDimArrayCstr* PD_PredictorGetOutputNames(PD_Predictor* pd_predictor);

// 获取输出 Tensor 数量
// 参数：pd_predictor - Predictor 输出tensor数量。
size_t PD_PredictorGetOutputNum(PD_Predictor* pd_predictor);

// 根据名称获取输出 Tensor 的句柄
// 参数：pd_predictor - Predictor 对象指针
//      name - Tensor 的名称
// 返回：PD_Tensor* - 指向 Tensor 的指针。
//      该 Tensor 需要显式调用`PD_TensorDestroy`来销毁。
PD_Tensor* PD_PredictorGetOutputHandle(PD_Predictor* pd_predictor, const char* name);
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

// 获取输入 tensor 的数量和名称
PD_OneDimArrayCstr* input_names = PD_PredictorGetInputNames(predictor);
printf("Input tensor number is: %d\n", input_names->size);
for(size_t index = 0; index < input_names->size; ++index) {
  printf("Input tensor %u name is: %s\n", index, input_names->data[index]);
}

// 根据名称获取第 0 个输入 Tensor
PD_Tensor* input_tensor = PD_PredictorGetInputHandle(predictor, input_names->data[0]);

// 获取输出 Tensor 的数量和名称
PD_OneDimArrayCstr* output_names = PD_PredictorGetOutputNames(predictor);
printf("Output tensor number is: %d\n", output_names->size);
for(size_t index = 0; index < output_names->size; ++index) {
  printf("Output tensor %u name is: %s\n", index, output_names->data[index]);
}

// 根据名称获取第 0 个输出 Tensor
PD_Tensor* output_tensor = PD_PredictorGetOutputHandle(predictor, output_names->data[0]);

// 销毁相应的对象
PD_TensorDestroy(output_tensor);
PD_OneDimArrayCstrDestroy(output_names);
PD_TensorDestroy(input_tensor);
PD_OneDimArrayCstrDestroy(input_names);
PD_PredictorDestroy(predictor);
```

## 执行预测

API 定义如下：

```c
// 执行模型预测，需要在设置输入Tensor数据后调用
// 参数：pd_predictor - Predictor 对象指针
// 返回：PD_Bool - 执行预测是否成功
PD_Bool PD_PredictorRun(PD_Predictor* pd_predictor);
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

// 获取输入 Tensor 并进行赋值
PD_OneDimArrayCstr* input_names = PD_PredictorGetInputNames(predictor);
PD_Tensor* input_tensor = PD_PredictorGetInputHandle(predictor, input_names->data[0]);
PD_TensorReshape(input_tensor, 4, input_shape);
PD_TensorCopyFromCpuFloat(input_tensor, input_data);

// 执行预测
PD_PredictorRun(pd_predictor);

// 获取预测输出 Tensor
PD_OneDimArrayCstr* output_names = PD_PredictorGetOutputNames(predictor);
PD_Tensor* output_tensor = PD_PredictorGetOutputHandle(predictor, output_names->data[0]);

// 获取输出 Tensor 数据
PD_OneDimArrayInt32* output_shape = PD_TensorGetShape(output_tensor);
int32_t out_size = 1;
for (size_t i = 0; i < output_shape->size; ++i) {
  out_size = out_size * output_shape->data[i];
}
float* out_data = (float*)malloc(out_size * sizeof(float));
PD_TensorCopyToCpuFloat(output_tensor, out_data);

// 销毁相关对象， 回收相关内存
free(out_data);
PD_OneDimArrayInt32Destroy(output_shape);
PD_TensorDestroy(output_tensor);
PD_OneDimArrayCstrDestroy(output_names);
PD_TensorDestroy(input_tensor);
PD_OneDimArrayCstrDestroy(input_names);
PD_PredictorDestroy(predictor);
free(input_data);
```