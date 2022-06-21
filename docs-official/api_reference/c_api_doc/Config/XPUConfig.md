# 使用 XPU 进行预测

API定义如下：

```c++
// 启用 XPU 进行预测
// 参数：pd_config         - Config 对象指针
//      l3_workspace_size - L3 cache 分配的显存大小
// 返回：None
void PD_ConfigEnableXpu(PD_Config* pd_config, int32_t l3_workspace_size);

// 判断是否启用 XPU 
// 参数：pd_config - Config 对象指针
// 返回：PD_Bool - 是否启用 XPU 
PD_Bool PD_ConfigUseXpu(PD_Config* pd_config);
```

代码示例：

```c
// 创建 Config 对象
PD_Config* config = PD_ConfigCreate();

// 启用 XPU，并设置 L3 cache 大小为 100MB
PD_ConfigEnableXpu(config, 100);

// 判断是否开启 XPU - True
printf("Use XPU is: %s\n", PD_ConfigUseXpu(config) ? "True" : "False");

// 销毁 Config 对象
PD_ConfigDestroy(config);
```
