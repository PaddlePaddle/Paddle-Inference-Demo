# GPU 混合精度推理

本文主要介绍 GPU 原生混合精度推理流程，主要分为模型精度转换和模型推理两个步骤。注意，该功能会在2.4发布，目前仅在 develop 分支上可使用。

- [1. 模型精度转换](#1)
- [2. 模型推理](#2)

<a name="1"></a>

## 1. 模型精度转换

`convert_to_mixed_precision`接口可对模型精度格式进行修改，可使用以下 python 脚本进行模型精度转换。

```python
from paddle.inference import PrecisionType, BackendType
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

实现思想基本可描述为：对模型进行拓扑遍历，依次检查算子在 backend 后端上是否支持 precision 精度类型，如果支持，则我们将该算子标记为混合精度类型，并且设置其输入输出均为 precision 类型，当算子输入与期望输入类型不符时，插入 `cast` 算子进行精度转换。


<a name="2"></a>

## 2. 模型推理

得到混合精度模型后，可按照正常的推理逻辑进行推理，详情请参考 [GPU 原生推理](https://www.paddlepaddle.org.cn/inference/master/guides/nv_gpu_infer/gpu_native_infer.html)。
