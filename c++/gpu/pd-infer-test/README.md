# pd-infer-test

pd-infer-test是一个快速使用Paddle Inference的命令行工具，你可以借助它快速实现：
- 使用随机生成的输入或者给定的输入快速测试模型性能
- 分析模型推理计算过程和输出结果是否准确

样例展示了 **Resnet50** 和 **yolov3** 在 GPU 下的推理过程。运行步骤如下：

## 一：获取 Paddle Inference 预测库

- [官网下载](https://www.paddlepaddle.org.cn/documentation/docs/zh/advanced_guide/inference_deployment/inference/build_and_install_lib_cn.html)
- 自行编译获取

将获取到的 Paddle Inference 预测库软链接或者重命名为 `paddle_inference`，并置于 `Paddle-Inference-Demo/c++/lib` 目录下。

## 二：获取模型

**Resnet50**: 点击[链接](https://paddle-inference-dist.bj.bcebos.com/Paddle-Inference-Demo/resnet50.tgz)下载模型

**yolov3**: 点击[链接](https://paddle-inference-dist.bj.bcebos.com/Paddle-Inference-Demo/yolov3_r50vd_dcn_270e_coco.tgz)下载模型

如果你想获取更多的**模型训练信息**，请访问[这里](https://github.com/PaddlePaddle/PaddleClas)。

## 三：编译样例
 
- 文件`pd-infer-test.cc` 为源码，目前使用`pd-infer-test`需要自行编译，后续版本会放入到官网预测库中。
- 脚本`compile.sh` 包含了第三方库、预编译库的信息配置。

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
TENSORRT_ROOT=/usr/local/TensorRT-8.4.3.1/
```

运行 `bash compile.sh` 编译样例。

## 四：运行样例

### **测试性能**

#### 使用原生 GPU 运行样例 (默认模式)
```shell
# resnet50
./build/pd-infer-test --model_file resnet50/inference.pdmodel --params_file resnet50/inference.pdiparams --warmup=100 --repeats=1000
# yolov3
./build/pd-infer-test --model_file yolov3_r50vd_dcn_270e_coco/model.pdmodel --params_file yolov3_r50vd_dcn_270e_coco/model.pdiparams --warmup=100 --repeats=1000
```
常用参数和作用
| 参数 | 作用 |
| --- | --- |
| --run_mode=trt_fp16 \| trt_fp32 | 使用 Trt dynamic shape fp16 \| fp32 运行样例 |
| --shapes="" | 指定输入shape, 可以通过指定shape设batch_size，例:--shapes="input1:1x32x32,input2:1x16"|
| --load_input="" | 指定输入文件 |
| --out_file="" | 保存输出到指定文件 |


### **调试精度**

```shell
# resnet50
./build/resnet50_test --model_file resnet50/inference.pdmodel --params_file resnet50/inference.pdiparams --run_mode=trt_fp32 --check_all=true
# yolov3
./build/resnet50_test --model_file resnet50/inference.pdmodel --params_file resnet50/inference.pdiparams --run_mode=trt_fp32 --check_all=true
```

常用参数和作用
| 参数 | 作用 |
| --- | --- |
| --baseline_mode=paddle_gpu \| trt_fp16 \| trt_fp32 | 指定精度检查的baseline模式 |
| --check=true | 只检查output |
| --check_all=true | 检查所有中间变量和output |
| --check_tensor="" | 检查指定tensor和output, 例: --check_tensor="tensor1,tensor2" |


## 更多链接
- [Paddle Inference使用Quick Start！](https://paddle-inference.readthedocs.io/en/latest/introduction/quick_start.html)
- [Paddle Inference C++ Api使用](https://paddle-inference.readthedocs.io/en/latest/api_reference/cxx_api_index.html)
- [Paddle Inference Python Api使用](https://paddle-inference.readthedocs.io/en/latest/api_reference/python_api_index.html)
