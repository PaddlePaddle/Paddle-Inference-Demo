# create_predictor 方法

API定义如下：

```python
# 根据 Config 构建预测执行器 Predictor
# 参数: config - 用于构建 Predictor 的配置信息
# 返回: Predictor - 预测执行器
paddle.inference.create_predictor(config: Config)
```

代码示例:

```python
# 引用 paddle inference 预测库
import paddle.inference as paddle_infer

# 创建 config
config = paddle_infer.Config("./mobilenet_v1.pdmodel", "./mobilenet_v1.pdiparams")

# 根据 config 创建 predictor
predictor = paddle_infer.create_predictor(config)
```

# get_version 方法

API定义如下：

```python
# 获取 Paddle 版本信息
# 参数: NONE
# 返回: str - Paddle 版本信息
paddle.inference.get_version()
```

代码示例:

```python
# 引用 paddle inference 预测库
import paddle.inference as paddle_infer

# 获取 Paddle 版本信息
paddle_infer.get_version()

# 获得输出如下:
# version: 2.0.0-rc0
# commit: 97227e6
# branch: HEAD
```

# convert_to_mixed_precision 方法

`convert_to_mixed_precision` 接口可对模型精度格式进行修改，如在选定的后端下，将 fp32 格式的模型转换为 fp16 格式，API 定义如下：

```python
# 模型精度转换
# 参数：model_file - fp32 模型文件路径
#      params_file - fp32 权重文件路径
#      mixed_model_file - 混合精度模型文件保存路径
#      mixed_params_file - 混合精度权重文件保存路径
#      mixed_precision - 转换精度，如 PrecisionType.kHalf
#      backend - 后端，如 PlaceType.kGPU
#      keep_io_types - 保留输入输出精度信息，若为 True 则输入输出保留为 fp32 类型，否则转为 precision 类型
#      black_list - 黑名单列表，哪些 op 不需要进行精度类型转换
# 返回：NONE
paddle.inference.convert_to_mixed_precision(
    model_file,
    params_file,
    mixed_model_file,
    mixed_params_file,
    mixed_precision,
    backend,
    keep_io_types,
    black_list)
```

代码示例：

```python
from paddle.inference import PrecisionType, PlaceType
from paddle.inference import convert_to_mixed_precision
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('src_model', type=str, help='src_model')
    parser.add_argument('src_params', type=str, help='src_params')
    parser.add_argument('dst_model', type=str, help='dst_model')
    parser.add_argument('dst_params', type=str, help='dst_params')
    args = parser.parse_args()
    black_list = set()

    convert_to_mixed_precision(
        args.src_model,     # fp32模型文件路径
        args.src_params,    # fp32权重文件路径
        args.dst_model,     # 混合精度模型文件保存路径
        args.dst_params,    # 混合精度权重文件保存路径
        PrecisionType.Half, # 转换精度，如 PrecisionType.Half
        PlaceType.GPU,      # 后端，如 PlaceType.GPU
        True,               # 保留输入输出精度信息，若为 True 则输入输出保留为 fp32 类型，否则转为 precision 类型
        black_list          # 黑名单列表，哪些 op 不需要进行精度类型转换
    )
```
