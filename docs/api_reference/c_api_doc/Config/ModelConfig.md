# 设置预测模型

## 从文件中加载预测模型 - 非Combined模型 

API定义如下：

```c
// 设置模型文件路径
// 参数：config - AnalysisConfig 对象指针
//      model_dir - 模型文件夹路径
//      params_path - NULL, 当输入模型为非 Combined 模型时，该参数为空指针
// 返回：None
void PD_SetModel(PD_AnalysisConfig* config, const char* model_dir, const char* params_path);

// 获取非combine模型的文件夹路径
// 参数：config - AnalysisConfig 对象指针
// 返回：const chart * - 模型文件夹路径
const char* PD_ModelDir(const PD_AnalysisConfig* config);
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

// 输出模型路径
printf("Non-combined model dir is: %s\n", PD_ModelDir(config));
```

## 从文件中加载预测模型 -  Combined 模型

API定义如下：

```c
// 设置模型文件路径
// 参数：config - AnalysisConfig 对象指针
//      model_dir - Combined 模型文件所在路径
//      params_path - Combined 模型参数文件所在路径
// 返回：None
void PD_SetModel(PD_AnalysisConfig* config, const char* model_dir, const char* params_path);

// 设置模型文件路径，当需要从磁盘加载 Combined 模型时使用。
// 参数：config - AnalysisConfig 对象指针
//      x - 模型文件路径
// 返回：None
void PD_SetProgFile(PD_AnalysisConfig* config, const char* x)

// 设置参数文件路径，当需要从磁盘加载 Combined 模型时使用
// 参数：config - AnalysisConfig 对象指针
//      x - 参数文件路径
// 返回：None
void PD_SetParamsFile(PD_AnalysisConfig* config, const char* x)

// 获取 Combined 模型的模型文件路径
// 参数：config - AnalysisConfig 对象指针
// 返回：const char* - 模型文件路径
const char* PD_ProgFile(const PD_AnalysisConfig* config)

// 获取 Combined 模型的参数文件路径
// 参数：config - AnalysisConfig 对象指针
// 返回：const char* - 参数文件路径
const char* PD_ParamsFile(const PD_AnalysisConfig* config)
```

代码示例 (1)：

```c
// 创建 AnalysisConfig 对象
PD_AnalysisConfig* config = PD_NewAnalysisConfig();

// 设置预测模型路径，这里为非 Combined 模型
const char* model_path  = "./model/model";
const char* params_path = "./model/params";
PD_SetModel(config, model_path, params_path);

// 根据 Config 创建 Predictor
PD_Predictor* predictor = PD_NewPredictor(config);

// 输出模型路径
printf("Non-combined model path is: %s\n", PD_ProgFile(config));
printf("Non-combined param path is: %s\n", PD_ParamsFile(config));
```

代码示例 (2)：

```c
// 创建 AnalysisConfig 对象
PD_AnalysisConfig* config = PD_NewAnalysisConfig();

// 设置预测模型路径，这里为非 Combined 模型
const char* model_path  = "./model/model";
const char* params_path = "./model/params";
PD_SetProgFile(config, model_path);
PD_SetParamsFile(config, params_path);

// 根据 Config 创建 Predictor
PD_Predictor* predictor = PD_NewPredictor(config);

// 输出模型路径
printf("Non-combined model path is: %s\n", PD_ProgFile(config));
printf("Non-combined param path is: %s\n", PD_ParamsFile(config));
```

## 从内存中加载预测模型

API定义如下：

```c
// 从内存加载模型
// 参数：config - AnalysisConfig 对象指针
//      prog_buffer - 内存中模型结构数据
//      prog_buffer_size - 内存中模型结构数据的大小
//      params_buffer - 内存中模型参数数据
//      params_buffer_size - 内存中模型参数数据的大小
// 返回：None
void PD_SetModelBuffer(PD_AnalysisConfig* config, 
                       const char* prog_buffer, size_t prog_buffer_size, 
                       const char* params_buffer, size_t params_buffer_size);


// 判断是否从内存中加载模型
// 参数：config - AnalysisConfig 对象指针
// 返回：bool - 是否从内存中加载模型
bool PD_ModelFromMemory(const PD_AnalysisConfig* config);
```

代码示例：

```c
// 定义文件读取函数
void read_file(const char * filename, char ** filedata, size_t * filesize) {
  FILE *file = fopen(filename, "rb");
  if (file == NULL) {
    printf("Failed to open file: %s\n", filename);
    return;
  }
  fseek(file, 0, SEEK_END);
  int64_t size = ftell(file);
  if (size == 0) {
    printf("File %s should not be empty, size is: %ld\n", filename, size);
    return;
  }
  rewind(file);
  *filedata = calloc(1, size+1);
  if (!(*filedata)) {
    printf("Failed to alloc memory.\n");
    return;
  }
  *filesize = fread(*filedata, 1, size, file);
  if ((*filesize) != size) {
    printf("Read binary file bytes do not match with fseek!\n");
    return;
  }
  fclose(file);
}

int main() {
  // 创建 AnalysisConfig 对象
  PD_AnalysisConfig* config = PD_NewAnalysisConfig();

  // 设置预测模型路径，这里为非 Combined 模型
  const char* model_path  = "./model/model";  
  const char* params_path = "./model/params";

  char * model_buffer = NULL;
  char * param_buffer = NULL;
  size_t model_size, param_size;
  read_file(model_path, &model_buffer, &model_size);
  read_file(params_path, &param_buffer, &param_size);

  if(model_buffer == NULL) {
    printf("Failed to load model buffer.\n");
    return 1;
  }
  if(param_buffer == NULL) {
    printf("Failed to load param buffer.\n");
    return 1;
  }

  // 加载模型文件到内存，并获取文件大小
  PD_SetModelBuffer(config, model_buffer, model_size, param_buffer, param_size);

  // 输出是否从内存中加载模型
  printf("Load model from memory is: %s\n", PD_ModelFromMemory(config) ? "true" : "false");

  // 根据 Config 创建 Predictor
  PD_Predictor* predictor = PD_NewPredictor(config);

  PD_DeletePredictor(predictor);
  PD_DeleteAnalysisConfig(config);

  free(model_buffer);
  free(param_buffer);
}
```
