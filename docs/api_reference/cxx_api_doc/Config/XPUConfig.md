
# 使用 XPU 进行预测

API定义如下：

```c++
// 启用 XPU 进行预测
// 参数：l3_workspace_size - l3 cache 分配的显存大小
// 返回：None
void EnableXpu(int l3_workspace_size = 0xfffc00);
```

代码示例：

```c++
// 创建 Config 对象
paddle_infer::Config config(FLAGS_model_dir);

// 启用 XPU，并设置 l3 cache 大小为 100M
config.EnableXpu(100);
```
