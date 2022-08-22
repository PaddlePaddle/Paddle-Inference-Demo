# 创建 Config

`Config` 对象相关方法用于创建预测相关配置，构建 `Predictor` 对象的配置信息，如模型路径、是否开启gpu等等。

相关方法定义如下：

```c
// 创建 Config 对象
// 参数：None
// 返回：PD_Config* - Config 对象指针
PD_Config* PD_ConfigCreate();

// 销毁 Config 对象
// 参数：pd_config - Config 对象指针
// 返回：None
void PD_ConfigDestroy(PD_Config* pd_config);
```

代码示例：

```c
// 创建 Config 对象
PD_Config* config = PD_ConfigCreate();

// 销毁 Config 对象
PD_ConfigDestroy(config);
```