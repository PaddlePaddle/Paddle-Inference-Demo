
# 使用 CustomDevice 进行预测

API定义如下：

```c++
// 启用 CustomDevice, device_type设备注册名，device_id设备ID
  void EnableCustomDevice(const std::string& device_type, int device_id = 0);
  
// CustomDevice启用混合精度
  void EnableCustomDeviceMixed(Precision precision_mode);

// 是否开启CustomDevice
  bool use_custom_device() const;


// 是否开启CustomDevice混合精度
  bool enable_custom_device_mixed() const;
  
```

代码示例：

```c++
// 创建 Config 对象
paddle_infer::Config config(FLAGS_model_dir);

// 开启 CustomDevice 模式，注册设备OpenCL
config.EnableCustomDevice("OpenCL", 0);

// 开启混合精度Half
config.EnableCustomDeviceMixed(paddle::AnalysisConfig::Precision::kHalf);

```

