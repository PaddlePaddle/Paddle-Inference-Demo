# 使用 XPU 进行预测

API定义如下：

```c++
// 启用 XPU 进行预测
// 参数：l3_size - l3 cache 分配的显存大小。注：昆仑1上最大为 16773120 Byte，昆仑2上最大为 67104768 Byte
// 参数：l3_locked - 分配的L3 cache是否可以锁定。如果为false，表示不锁定L3 cache，则分配的L3 cache可以多个模型共享，多个共享L3 cache的模型在卡上将顺序执行
// 参数：conv_autotune - 是否对模型中的conv算子进行autotune。如果为true，则在第一次执行到某个维度的conv算子时，将自动搜索更优的算法，用以提升后续相同维度的conv算子的性能
// 参数：conv_autotune_file - 指定autotune文件路径。如果指定autotune_file，则使用文件中指定的算法，不再重新进行autotune
// 参数：transformer_encoder_precision - multi_encoder的计算精度
// 参数：transformer_encoder_adaptive_seqlen - multi_encoder的输入是否可变长
// 参数：enable_multi_stream - 是否启用多流推理，如果启动，将自动创建新的流用于推理
// 返回：None
void PD_ConfigEnableXpu(__pd_keep PD_Config* pd_config,
                        int32_t l3_size,
                        PD_Bool l3_locked,
                        PD_Bool conv_autotune,
                        const char* conv_autotune_file,
                        const char* transformer_encoder_precision,
                        PD_Bool transformer_encoder_adaptive_seqlen,
                        PD_Bool enable_multi_stream);

// 判断是否启用 XPU 
// 参数：pd_config - Config 对象指针
// 返回：PD_Bool - 是否启用 XPU 
PD_Bool PD_ConfigUseXpu(PD_Config* pd_config);
```

代码示例：

```c
// 创建 Config 对象
PD_Config* config = PD_ConfigCreate();

// 启用 XPU，并设置 l3 cache 大小为 10M
PD_ConfigEnableXpu(config, 10*1024*1024);

// 判断是否开启 XPU - True
printf("Use XPU is: %s\n", PD_ConfigUseXpu(config) ? "True" : "False");

// 销毁 Config 对象
PD_ConfigDestroy(config);
```
