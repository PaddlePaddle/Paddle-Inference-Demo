# 使用 IPU 进行预测

API定义如下：

```c++
// 启用 IPU 进行预测
// 参数：ipu_device_num - 所需要的IPU个数.
// 参数：ipu_micro_batch_size - 计算图输入的batch size，用于根据输入batch size进行全图Tensor shape推导，仅在动态输入batch size的情况生效
// 参数：ipu_enable_pipelining - 使能IPU间数据流水
// 参数：ipu_batches_per_step - 在使能数据流水的条件下，指定每次跑多少batch的数据，如果关闭数据流水，该值应设置为1
// 返回：None
void EnableIpu(int ipu_device_num = 1, int ipu_micro_batch_size = 1,
               bool ipu_enable_pipelining = false,
               int ipu_batches_per_step = 1);

// 配置 IPU 构图参数
// 参数：ipu_enable_fp16 - 使能float16模式，将float32计算图转换为float16计算图.
// 参数：ipu_replica_num - 设置实例个数，举例ipu_device_num=2，表示单个实例需要2个IPU运行，设置ipu_replica_num=8,表示总共有8个相同实例，所以总共需要16个IPU.
// 参数：ipu_available_memory_proportion - 设置matmul/conv Op可使用的内存比例，取值(0.0, 1.0], 比例越高，计算性能越好.
// 参数：ipu_enable_half_partial - matmul Op中间结果以float16存储于片上.
// 返回：None
void SetIpuConfig(bool ipu_enable_fp16 = false, int ipu_replica_num = 1,
                  float ipu_available_memory_proportion = 1.0,
                  bool ipu_enable_half_partial = false);
```

代码示例：

```c++
// 创建 Config 对象
paddle_infer::Config config(FLAGS_model_dir);

// 启用 IPU，并设置单个实例所需要的IPU个数为1
config.EnableIpu(1);
// 使能float16模式
config.SetIpuConfig(true);
```
