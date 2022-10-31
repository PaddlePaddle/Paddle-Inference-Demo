
# 使用 XPU 进行预测

API定义如下：

```c++
// 启用 XPU 进行预测
// 参数：l3_workspace_size - l3 cache 分配的显存大小，最大为16M
// 参数：locked - 分配的L3 cache是否可以锁定。如果为false，表示不锁定L3 cache，则分配的L3 cache可以多个模型共享，多个共享L3 cache的模型在卡上将顺序执行
// 参数：autotune - 是否对模型中的conv算子进行autotune。如果为true，则在第一次执行到某个维度的conv算子时，将自动搜索更优的算法，用以提升后续相同维度的conv算子的性能
// 参数：autotune_file - 指定autotune文件路径。如果指定autotune_file，则使用文件中指定的算法，不再重新进行autotune
// 参数：precision - multi_encoder的计算精度
// 参数：adaptive_seqlen - multi_encoder的输入是否可变长
// 参数：enable_multi_stream - 是否开启XPU的multi stream功能
// 返回：None
void EnableXpu(int l3_workspace_size = 0xfffc00,
               bool locked = false,
               bool autotune = true,
               const std::string& autotune_file = "",
               const std::string& precision = "int16",
               bool adaptive_seqlen = false,
               bool enable_multi_stream = false);
```

代码示例：

```c++
// 创建 Config 对象
paddle_infer::Config config(FLAGS_model_dir);

// 开启 Lite 子图引擎
config.EnableLiteEngine();

// 启用 XPU，并设置 l3 cache 大小为 10M
config.EnableXpu(10*1024*1024);
```

**注意事项：**
Xpu 推理依赖 Lite 子图引擎，配置需开启 EnableLiteEngine，API 文档参考[设置模型优化方法](./OptimConfig.md)。
