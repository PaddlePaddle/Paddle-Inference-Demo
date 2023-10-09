# 运行 ERNIE 变长推理样例

ernie_varlen 样例展示了单输入模型在 GPU 下的推理过程。运行步骤如下：
可直接运行：sh run.sh

## 一：准备环境

请您在环境中安装2.0或以上版本的 Paddle，具体的安装方式请参照[飞桨官方页面](https://www.paddlepaddle.org.cn/)的指示方式。


## 二：获取 Ernie 模型

点击[链接](http://paddle-inference-dist.bj.bcebos.com/tensorrt_test/ernie_model_4.tar.gz)下载模型， 如果你想获取更多的**Ernie模型信息**，请访问[这里](https://www.paddlepaddle.org.cn/paddle/ernie)。


## 三：运行预测

- 文件`infer_ernie_varlen.py` 包含了创建predictor，推理，获取输出的等功能。

### 使用 Trt Fp16 运行样例

```shell
python infer_ernie_varlen.py --model_dir=./ernie_model_4/  --run_mode=trt_fp16
```

## 相关链接
- [Paddle Inference Python Api使用](https://paddle-inference.readthedocs.io/en/latest/api_reference/python_api_index.html)

