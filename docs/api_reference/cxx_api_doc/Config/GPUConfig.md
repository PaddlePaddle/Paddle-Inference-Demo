# 使用 GPU 进行预测

**注意：**
1. Config 默认使用 CPU 进行预测，需要通过 `EnableUseGpu` 来启用 GPU 预测
2. 可以尝试启用 CUDNN 和 TensorRT 进行 GPU 预测加速

## GPU 设置

API定义如下：

```c++
// 启用 GPU 进行预测
// 参数：memory_pool_init_size_mb - 初始化分配的gpu显存，以MB为单位
//      device_id - 设备id
// 返回：None
void EnableUseGpu(uint64_t memory_pool_init_size_mb, int device_id = 0);

// 禁用 GPU 进行预测
// 参数：None
// 返回：None
void DisableGpu();

// 判断是否启用 GPU 
// 参数：None
// 返回：bool - 是否启用 GPU 
bool use_gpu() const;

// 获取 GPU 的device id
// 参数：None
// 返回：int -  GPU 的device id
int gpu_device_id() const;

// 获取 GPU 的初始显存大小
// 参数：None
// 返回：int -  GPU 的初始的显存大小
int memory_pool_init_size_mb() const;

// 初始化显存占总显存的百分比
// 参数：None
// 返回：float - 初始的显存占总显存的百分比
float fraction_of_gpu_memory_for_pool() const;

// 开启线程流，目前的行为是为每一个线程绑定一个流，在将来该行为可能改变
// 参数：None
// 返回：None
void EnableGpuMultiStream();

// 判断是否开启线程流
// 参数：None
// 返回：bool - 是否是否开启线程流
bool thread_local_stream_enabled() const;

// 启用 GPU FP16 计算精度进行预测
// 参数：op_list - 保持 FP32 计算精度算子名单
// 返回：None
void Exp_EnableUseGpuFp16(std::unordered_set<std::string> op_list = {});

// 判断是否启用 GPU FP16 计算精度 
// 参数：None
// 返回：bool - 是否启用 GPU FP16 计算精度
bool gpu_fp16_enabled() const;
```

GPU设置代码示例：

```c++
// 创建默认 Config 对象
paddle_infer::Config config;

// 启用 GPU 进行预测 - 初始化 GPU 显存 100M, Deivce_ID 为 0
config.EnableUseGpu(100, 0);
// 通过 API 获取 GPU 信息
std::cout << "Use GPU is: " << config.use_gpu() << std::endl; // true
std::cout << "Init mem size is: " << config.memory_pool_init_size_mb() << std::endl;
std::cout << "Init mem frac is: " << config.fraction_of_gpu_memory_for_pool() << std::endl;
std::cout << "GPU device id is: " << config.gpu_device_id() << std::endl;

// 禁用 GPU 进行预测
config.DisableGpu();
// 通过 API 获取 GPU 信息
std::cout << "Use GPU is: " << config.use_gpu() << std::endl; // false

// 启用 GPU FP16 计算精度进行预测
config.EnableUseGpu(100, 0);
config.Exp_EnableUseGpuFp16();
// 通过 API 获取是否启用了 GPU FP16 计算精度
std::cout << "Use GPU FP16 is: " << config.gpu_fp16_enabled() << std::endl; // true
```

开启多线程流代码示例：

```c++
// 自定义 Barrier 类，用于线程间同步
class Barrier {
 public:
  explicit Barrier(std::size_t count) : _count(count) {}
  void Wait() {
    std::unique_lock<std::mutex> lock(_mutex);
    if (--_count) {
      _cv.wait(lock, [this] { return _count == 0; });
    } else {
      _cv.notify_all();
    }
  }
 private:
  std::mutex _mutex;
  std::condition_variable _cv;
  std::size_t _count;
};

int test_main(const paddle_infer::Config& config, Barrier* barrier = nullptr) {
  static std::mutex mutex;
  // 创建 Predictor 对象
  std::shared_ptr<paddle_infer::Predictor> predictor;
  {
    std::unique_lock<std::mutex> lock(mutex);
    predictor = std::move(paddle_infer::CreatePredictor(config));
  }
  if (barrier) {
    barrier->Wait();
  }
  // 准备输入数据
  int input_num = shape_production(INPUT_SHAPE);
  std::vector<float> input_data(input_num, 1);
  auto input_names = predictor->GetInputNames();
  auto input_tensor = predictor->GetInputHandle(input_names[0]);
  input_tensor->Reshape(INPUT_SHAPE);
  input_tensor->CopyFromCpu(input_data.data());
  // 执行预测
  predictor->Run();
  // 获取预测输出
  auto output_names = predictor->GetOutputNames();
  auto output_tensor = predictor->GetOutputHandle(output_names[0]);
  std::vector<int> output_shape = output_tensor->shape();
  std::cout << "Output shape is " << shape_to_string(output_shape) << std::endl;
}

int main(int argc, char **argv) {
  const size_t thread_num = 5;
  std::vector<std::thread> threads(thread_num);
  Barrier barrier(thread_num);
  // 创建 5 个线程，并为每个线程开启一个单独的GPU Stream
  for (size_t i = 0; i < threads.size(); ++i) {
    threads[i] = std::thread([&barrier, i]() {
      paddle_infer::Config config;
      config.EnableUseGpu(100, 0);
      config.SetModel(FLAGS_infer_model);
      config.EnableGpuMultiStream();
      test_main(config, &barrier);
    });
  }
  for (auto& th : threads) {
    th.join();
  }
}
```

## CUDNN 设置

**注意：** 启用 CUDNN 的前提为已经启用 GPU，否则启用 CUDNN 无法生效。

API定义如下：

```c++
// 启用 CUDNN 进行预测加速
// 参数：None
// 返回：None
void EnableCUDNN();

// 判断是否启用 CUDNN 
// 参数：None
// 返回：bool - 是否启用 CUDNN
bool cudnn_enabled() const;
```

代码示例：

```c++
// 创建默认 Config 对象
paddle_infer::Config config();

// 启用 GPU 进行预测
config.EnableUseGpu(100, 0);
// 启用 CUDNN 进行预测加速
config.EnableCUDNN();
// 通过 API 获取 CUDNN 启用结果
std::cout << "Enable CUDNN is: " << config.cudnn_enabled() << std::endl; // true

// 禁用 GPU 进行预测
config.DisableGpu();
// 启用 CUDNN 进行预测加速 - 因为 GPU 被禁用，因此 CUDNN 启用不生效
config.EnableCUDNN();
// 通过 API 获取 CUDNN 启用结果
std::cout << "Enable CUDNN is: " << config.cudnn_enabled() << std::endl; // false
```

## TensorRT 设置

**注意：** 
1. 启用 TensorRT 的前提为已经启用 GPU，否则启用 TensorRT 无法生效
2. 对存在LoD信息的模型，如Bert, Ernie等NLP模型，必须使用动态 Shape
3. 启用 TensorRT OSS 可以支持更多 plugin，详细参考 [TensorRT OSS](https://news.developer.nvidia.com/nvidia-open-sources-parsers-and-plugins-in-tensorrt/)

更多 TensorRT 详细信息，请参考 [使用Paddle-TensorRT库预测](../../../optimize/paddle_trt)。

API定义如下：

```c++
// 启用 TensorRT 进行预测加速
// 参数：workspace_size     - 指定 TensorRT 使用的工作空间大小
//      max_batch_size     - 设置最大的 batch 大小，运行时 batch 大小不得超过此限定值
//      min_subgraph_size  - Paddle-TRT 是以子图的形式运行，为了避免性能损失，当子图内部节点个数
//                           大于 min_subgraph_size 的时候，才会使用 Paddle-TRT 运行
//      precision          - 指定使用 TRT 的精度，支持 FP32(kFloat32)，FP16(kHalf)，Int8(kInt8)
//      use_static         - 若指定为 true，在初次运行程序的时候会将 TRT 的优化信息进行序列化到磁盘上，
//                           下次运行时直接加载优化的序列化信息而不需要重新生成
//      use_calib_mode     - 若要运行 Paddle-TRT INT8 离线量化校准，需要将此选项设置为 true
// 返回：None
void EnableTensorRtEngine(int workspace_size = 1 << 20,
                          int max_batch_size = 1, int min_subgraph_size = 3,
                          Precision precision = Precision::kFloat32,
                          bool use_static = false,
                          bool use_calib_mode = true);
// 判断是否启用 TensorRT 
// 参数：None
// 返回：bool - 是否启用 TensorRT
bool tensorrt_engine_enabled() const;

// 设置 TensorRT 的动态 Shape
// 参数：min_input_shape          - TensorRT 子图支持动态 shape 的最小 shape
//      max_input_shape          - TensorRT 子图支持动态 shape 的最大 shape
//      optim_input_shape        - TensorRT 子图支持动态 shape 的最优 shape
//      disable_trt_plugin_fp16  - 设置 TensorRT 的 plugin 不在 fp16 精度下运行
// 返回：None
void SetTRTDynamicShapeInfo(
      std::map<std::string, std::vector<int>> min_input_shape,
      std::map<std::string, std::vector<int>> max_input_shape,
      std::map<std::string, std::vector<int>> optim_input_shape,
      bool disable_trt_plugin_fp16 = false);

// 启用 TensorRT OSS 进行预测加速
// 参数：None
// 返回：None
void EnableTensorRtOSS();

// 判断是否启用 TensorRT OSS
// 参数：None
// 返回：bool - 是否启用 TensorRT OSS
bool tensorrt_oss_enabled();

/// 启用TensorRT DLA进行预测加速
/// 参数：dla_core - DLA设备的id，可选0，1，...，DLA设备总数 - 1
/// 返回：None
void EnableTensorRtDLA(int dla_core = 0);

/// 判断是否已经开启TensorRT DLA加速
/// 参数：None
/// 返回：bool - 是否已开启TensorRT DLA加速
bool tensorrt_dla_enabled();
```

代码示例 (1)：使用 TensorRT FP32 / FP16 / INT8 进行预测

```c++
// 创建 Config 对象
paddle_infer::Config config(FLAGS_infer_model + "/mobilenet");

// 启用 GPU 进行预测
config.EnableUseGpu(100, 0);

// 启用 TensorRT 进行预测加速 - FP32
config.EnableTensorRtEngine(1 << 20, 1, 3, 
                            paddle_infer::PrecisionType::kFloat32, false, false);
// 通过 API 获取 TensorRT 启用结果 - true
std::cout << "Enable TensorRT is: " << config.tensorrt_engine_enabled() << std::endl;

// 启用 TensorRT 进行预测加速 - FP16
config.EnableTensorRtEngine(1 << 20, 1, 3, 
                            paddle_infer::PrecisionType::kHalf, false, false);
// 通过 API 获取 TensorRT 启用结果 - true
std::cout << "Enable TensorRT is: " << config.tensorrt_engine_enabled() << std::endl;

// 启用 TensorRT 进行预测加速 - Int8
config.EnableTensorRtEngine(1 << 20, 1, 3, 
                            paddle_infer::PrecisionType::kInt8, false, true);
// 通过 API 获取 TensorRT 启用结果 - true
std::cout << "Enable TensorRT is: " << config.tensorrt_engine_enabled() << std::endl;
```

代码示例 (2)：使用 TensorRT 动态 Shape 进行预测

```c++
// 创建 Config 对象
paddle_infer::Config config(FLAGS_infer_model + "/mobilenet");

// 启用 GPU 进行预测
config.EnableUseGpu(100, 0);

// 启用 TensorRT 进行预测加速 - Int8
config.EnableTensorRtEngine(1 << 30, 1, 1,
                            paddle_infer::PrecisionType::kInt8, false, true);
// 设置模型输入的动态 Shape 范围
std::map<std::string, std::vector<int>> min_input_shape = {{"image", {1, 1, 3, 3}}};
std::map<std::string, std::vector<int>> max_input_shape = {{"image", {1, 1, 10, 10}}};
std::map<std::string, std::vector<int>> opt_input_shape = {{"image", {1, 1, 3, 3}}};
// 设置 TensorRT 的动态 Shape
config.SetTRTDynamicShapeInfo(min_input_shape, max_input_shape, opt_input_shape);
```

代码示例 (3)：使用 TensorRT OSS 进行预测

```c++
// 创建 Config 对象
paddle_infer::Config config(FLAGS_infer_model + "/mobilenet");

// 启用 GPU 进行预测
config.EnableUseGpu(100, 0);

// 启用 TensorRT 进行预测加速
config.EnableTensorRtEngine();
// 启用 TensorRT OSS 进行预测加速
config.EnableTensorRtOSS();

// 通过 API 获取 TensorRT OSS 启用结果 - true
std::cout << "Enable TensorRT is: " << config.tensorrt_oss_enabled() << std::endl;
```
