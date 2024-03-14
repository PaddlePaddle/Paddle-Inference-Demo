# paddle-TRT 精度调试运行示例

精度调试运行示例以 **Resnet50** 为例展示了使用随机生成的输入分析模型中间变量及结果与baseline的精度差异，运行步骤如下：

## 一：获取 Paddle Inference 预测库

- [官网下载](https://www.paddlepaddle.org.cn/documentation/docs/zh/advanced_guide/inference_deployment/inference/build_and_install_lib_cn.html)（2.6及以上）
- 自行编译获取

将获取到的 Paddle Inference 预测库软链接或者重命名为 `paddle_inference`，并置于 `Paddle-Inference-Demo/c++/lib` 目录下。

## 二：获取 Resnet50 模型

点击[链接](https://paddle-inference-dist.bj.bcebos.com/Paddle-Inference-Demo/resnet50.tgz)下载模型。如果你想获取更多的**模型训练信息**，请访问[这里](https://github.com/PaddlePaddle/PaddleClas)。

## 三：编译样例
 
- 文件`tensorrt_precision_debug.cc` 为预测的样例程序（程序中的输入为随机生成的取值范围 0～1 的 tensor，如果您有 opencv 或其他方式进行数据读取的需求，需要对程序进行一定的修改）。    
- 脚本`compile.sh` 包含了第三方库、预编译库的信息配置。  
- 脚本`run.sh` 为一键运行脚本。

编译前，需要根据自己的环境修改 `compile.sh` 中的相关代码配置依赖库：
```shell
# 根据预编译库中的version.txt信息判断是否将以下三个标记打开
WITH_MKL=ON
WITH_GPU=ON
USE_TENSORRT=ON

# 配置预测库的根目录
LIB_DIR=${work_path}/../lib/paddle_inference

# 如果上述的WITH_GPU 或 USE_TENSORRT设为ON，请设置对应的CUDA， CUDNN， TENSORRT的路径。
CUDNN_LIB=/usr/lib/x86_64-linux-gnu/
CUDA_LIB=/usr/local/cuda/lib64
TENSORRT_ROOT=/usr/local/TensorRT-8.6.1.6/
```

运行 `bash compile.sh` 编译样例。

## 四：运行样例

```shell
./build/tensorrt_precision_debug --run_mode=trt_fp16 --model_file resnet50/inference.pdmodel --params_file resnet50/inference.pdiparams
```

运行成功后在终端按照网络算子执行顺序打印出所有中间变量的值，以及与baseline的精度差异，示例如下：

| Operator Type | Tensor Name | Shape | Mismatched Elements| Max Atol | Max Rtol| Min Val(base) | Max Val(base) |
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| batch_norm | batch_norm_0.tmp_2 | match, [1,64,112,112] | 0/802816 | 0.000001 | 0.021647 | -1.157186(-1.157185) | 1.551287(1.551287) |
| ... | ... | ... | ... | ... | ... | ... | ... |


baseline为关掉IR优化的Paddle GPU FP32的预测结果。
其中 Max Atol 和 Max Rtol 分别为最大绝对误差和最大相对误差，Min Val(base) 和 Max Val(base) 分别为最小值和最大值，括号内为baseline的最大值和最小值。最大值与最小值基本和baseline一致时，说明精度差异较小，建议用数据集检查精度差异。
## 五：进阶调试

本脚本通过 `save_baseline_hook` 保存 baseline 的结果，通过 `check_diff::mark_tensor_names` 指定需要检查的中间变量, 并通过 `assert_tensor_close_hook` 检查中间变量和结果的精度差异。

### 1) 调整检查的中间变量

检查所有中间变量 (默认)
```c++
check_diff::mark_tensor_names = check_diff::tensor_names;
```
`check_diff::tensor_names` 检查所有中间变量会影响 trt 的融合策略，可能会导致调试和实际运行时结果不一致。另外在模型较大时，检查所有中间变量会占用大量内存和显存。

只检查trt融合后的 layer output 可以不破坏trt融合策略，通过 Inspector 可将 trt 融合后的 layer 信息保存到 cache 文件夹中的 `engine_info_` 开头的 json 文件中：
```c++
config.EnableTensorRtInspector(true);
```
可以通过如下脚本从 `engine_info_xxx.json` 中获取需要检查的中间变量名：

```python
import json

check_diff_mark_tensor_names = []

with open("cache/engine_info_xxx.json", "r") as f:
    data = json.load(f)
for layer in data:
    for output in layer['Outputs']:
        output_name = output["Name"]
        if "subgraph" in output_name:
            output_name = output_name[0:output_name.index("_subgraph")]
            check_diff_mark_tensor_names.append(output_name)
```
`save_baseline_hook` 保存的 tensor_name 是算子执行顺序，因此可以从中截取出网络中某一段的tensor_name，然后将其加入到 `check_diff::mark_tensor_names` 中，在较大模型的检查中可以节省内存和显存资源。

### 2) 更换baseline
baseline 的 predictor 通过 `InitBaselinePredictor` 创建，默认使用关掉 IR 优化的 Paddle GPU 的 FP32 预测结果，此时保存的 tensor_name 是算子执行顺序，更换 baseline 可以通过创建新的 predictor 实现。


## 更多链接
- [Paddle Inference使用Quick Start！](https://www.paddlepaddle.org.cn/inference/master/guides/quick_start/index_quick_start.html)
- [Paddle Inference C++ Api使用](https://www.paddlepaddle.org.cn/inference/master/api_reference/cxx_api_doc/cxx_api_index.html)
- [Paddle Inference Python Api使用](https://www.paddlepaddle.org.cn/inference/master/api_reference/python_api_doc/python_api_index.html)
