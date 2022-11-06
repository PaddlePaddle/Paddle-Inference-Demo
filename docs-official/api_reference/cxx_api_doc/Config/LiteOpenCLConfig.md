
# 使用 Lite OpenCL 进行预测

API定义如下：

```c++
// 启用 Lite模式下的OpenCL加速
void EnableOpenCL();

// 是否开启Lite模式的OpenCL
bool use_opencl() const;
```

代码示例：

```c++
// 创建 Config 对象
paddle_infer::Config config(FLAGS_model_dir);

// 开启 Lite 模式
config.EnableLiteEngine();

// 启用 OpenCL 加速
config.EnableOpenCL();
```

**注意事项：**
Paddle Inference下的Lite模式，默认使用CPU进行运算，EnableOpenCL可以开启Lite的OpenCL模式进行GPU加速。