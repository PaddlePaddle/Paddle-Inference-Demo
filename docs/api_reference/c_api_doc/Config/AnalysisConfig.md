# 创建 AnalysisConfig

`AnalysisConfig` 对象相关方法用于创建预测相关配置，构建 `Predictor` 对象的配置信息，如模型路径、是否开启gpu等等。

相关方法定义如下：

```c
// 创建 AnalysisConfig 对象
// 参数：None
// 返回：PD_AnalysisConfig* - AnalysisConfig 对象指针
PD_AnalysisConfig* PD_NewAnalysisConfig();

// 删除 Config 对象
// 参数：config - AnalysisConfig 对象指针
// 返回：None
void PD_DeleteAnalysisConfig(PD_AnalysisConfig* config);

// 设置 Config 为无效状态，仅内部使用，保证每一个 Config 仅用来初始化一次 Predictor
// 参数：config - AnalysisConfig 对象指针
// 返回：None
void PD_SetInValid(PD_AnalysisConfig* config);

// 判断当前 Config 是否有效
// 参数：config - AnalysisConfig 对象指针
// 返回：bool - 当前 Config 是否有效
bool PD_IsValid(const PD_AnalysisConfig* config);
```

代码示例：

```c
// 创建 AnalysisConfig 对象
PD_AnalysisConfig* config = PD_NewAnalysisConfig();

// 判断当前 Config 是否有效 - True
printf("Config validation is: %s\n", PD_IsValid(config) ? "True" : "False");

// 设置 Config 为无效状态
PD_SetInValid(config);

// 判断当前 Config 是否有效 - false
printf("Config validation is: %s\n", PD_IsValid(config) ? "True" : "False");

// 删除 AnalysisConfig 对象
PD_DeleteAnalysisConfig(config);
```