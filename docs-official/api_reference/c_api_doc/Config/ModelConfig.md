# 设置预测模型

## 从文件中加载预测模型

API定义如下：

```c
// 设置模型文件路径
// 参数：pd_config        - Config 对象指针
//      prog_file_path   - 模型文件所在路径
//      params_file_path - 模型参数文件所在路径
// 返回：None
void PD_ConfigSetModel(PD_Config* pd_config, const char* prog_file_path, const char* params_file_path);

// 设置模型文件路径，当需要从磁盘加载模型时使用。
// 参数：pd_config      - Config 对象指针
//      prog_file_path - 模型文件路径
// 返回：None
void PD_ConfigSetProgFile(PD_Config* pd_config, const char* prog_file_path);


// 设置参数文件路径，当需要从磁盘加载模型时使用
// 参数：pd_config        - Config 对象指针
//      params_file_path - 参数文件路径
// 返回：None
void PD_ConfigSetParamsFile(PD_Config* pd_config, const char* params_file_path);

// 获取 Combined 模型的模型文件路径
// 参数：pd_config - Config 对象指针
// 返回：const char* - 模型文件路径
const char* PD_ConfigGetProgFile(PD_Config* pd_config);

// 获取 Combined 模型的参数文件路径
// 参数：pd_config - Config 对象指针
// 返回：const char* - 参数文件路径
const char* PD_ConfigGetParamsFile(PD_Config* pd_config);
```

代码示例 (1)：

```c
// 创建 Config 对象
PD_Config* config = PD_ConfigCreate();

// 设置预测模型路径，这里为 Combined 模型
const char* model_path  = "./model/inference.pdmodel";  
const char* params_path = "./model/inference.pdiparams";
PD_ConfigSetModel(config, model_path, params_path);

// 输出模型路径
printf("Non-combined model path is: %s\n", PD_ConfigGetProgFile(config));
printf("Non-combined param path is: %s\n", PD_ConfigGetParamsFile(config));

// 销毁 Config 对象
PD_ConfigDestroy(config);
```

代码示例 (2)：

```c
// 创建 Config 对象
PD_Config* config = PD_ConfigCreate();

// 设置预测模型路径，这里为 Combined 模型
const char* model_path  = "./model/inference.pdmodel";  
const char* params_path = "./model/inference.pdiparams";
PD_ConfigSetProgFile(config, model_path);
PD_ConfigSetParamsFile(config, params_path);

// 输出模型路径
printf("Non-combined model path is: %s\n", PD_ConfigGetProgFile(config));
printf("Non-combined param path is: %s\n", PD_ConfigGetParamsFile(config));

// 销毁 Config 对象
PD_ConfigDestroy(config);
```

## 从内存中加载预测模型

API定义如下：

```c
// 从内存加载模型
// 参数：pd_config          - Config 对象指针
//      prog_buffer        - 内存中模型结构数据
//      prog_buffer_size   - 内存中模型结构数据的大小
//      params_buffer      - 内存中模型参数数据
//      params_buffer_size - 内存中模型参数数据的大小
// 返回：None
void PD_ConfigSetModelBuffer(PD_Config* pd_config,
                             const char* prog_buffer, size_t prog_buffer_size,
                             const char* params_buffer, size_t params_buffer_size);

// 判断是否从内存中加载模型
// 参数：pd_config - Config 对象指针
// 返回：PD_Bool - 是否从内存中加载模型
PD_Bool PD_ConfigModelFromMemory(PD_Config* pd_config);
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
  // 创建 Config 对象
  PD_Config* config = PD_ConfigCreate();

  // 设置推理模型路径
  const char* model_path  = "./model/inference.pdmodel";  
  const char* params_path = "./model/inference.pdiparams";

  // 加载模型文件到内存，并获取文件大小
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

  // 从内存中加载模型
  PD_ConfigSetModelBuffer(config, model_buffer, model_size, param_buffer, param_size);

  // 输出是否从内存中加载模型
  printf("Load model from memory is: %s\n", PD_ConfigModelFromMemory(config) ? "true" : "false");

 // 销毁 Config 对象
  PD_ConfigDestroy(config);

  free(model_buffer);
  free(param_buffer);
}
```
