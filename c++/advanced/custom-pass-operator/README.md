# 运行自定义Pass调用自定义算子示例

基于`ResNet50`模型，我们实现一个自定义`Pass`将模型中的`Paddle`原生`pd_op.relu`算子替换成自定义`custom_op.custom_relu`算子。

## 一：获取 Paddle Inference 预测库

- [官网下载](https://www.paddlepaddle.org.cn/documentation/docs/zh/advanced_guide/inference_deployment/inference/build_and_install_lib_cn.html)
- 自行编译获取

将获取到的 Paddle Inference 预测库软链接或者重命名为 `paddle_inference`，并置于 `Paddle-Inference-Demo/c++/lib` 目录下。

## 二：获取 Resnet50 模型

点击[链接](https://paddle-inference-dist.bj.bcebos.com/Paddle-Inference-Demo/resnet50.tgz)下载模型。如果你想获取更多的**模型训练信息**，请访问[这里](https://github.com/PaddlePaddle/PaddleClas)。

## 三：编译样例

- 文件`custom_pass_test.cc` 为预测的样例程序（程序中的输入为固定值）。
- 脚本`compile.sh` 包含了第三方库、预编译库的信息配置。
- 脚本`run.sh` 为一键运行脚本。

编译前，需要根据自己的环境修改 `compile.sh` 中的相关代码配置依赖库：
```shell
# 编译的 demo 名称
DEMO_NAME=custom_pass_test

# 根据预编译库中的version.txt信息判断是否将以下三个标记打开
WITH_MKL=ON
WITH_GPU=ON

#
# 自定义Pass相关头文件需要c++17标准来编译
#

# 配置预测库的根目录
LIB_DIR=${work_path}/../lib/paddle_inference

# 需要编译的自定义算子文件
CUSTOM_OPERATOR_FILES="custom_relu_op_pass/custom_relu_op.cc;custom_relu_op_pass/custom_relu_op.cu;"
# 需要编译的自定义pass文件
CUSTOM_PASS_FILES="custom_relu_op_pass/custom_relu_pass.cc"
```

运行 `bash compile.sh` 编译样例。

## 四：运行样例
```shell
./build/custom_pass_test --model_file resnet50/inference.pdmodel --params_file resnet50/inference.pdiparams
```

作为调试，如果你开启`config.SwitchIrDebug()`，会看到跑完Pass前后的日志，可以进一步观察到模型中的`pd_op.relu`被`custom_op.custom_relu`替换了。
```
[1m[35m--- Running PIR pass [relu_replace_pass][0m
===--------------------------------------------------------------------===
        IRPrinting on builtin.module before relu_replace_pass pass
===--------------------------------------------------------------------===
...
(%375) = "pd_op.relu" (%369) {is_persistable:[false],stop_gradient:[false]} : (pd_op.tensor<-1x128x28x28xf32>) -> pd_op.tensor<-1x128x28x28xf32>
...

===-------------------------------------------------------------------===
        IRPrinting on builtin.module after relu_replace_pass pass
===-------------------------------------------------------------------===
...
(%375) = "custom_op.custom_relu" (%369) {stop_gradient:[false]} : (pd_op.tensor<-1x128x28x28xf32>) -> pd_op.tensor<-1x128x28x28xf32>
...
```
整个程序运行结束后，程序会将模型输出结果打印到屏幕，并且精度也是正确的。

## 更多链接
- [Paddle Inference使用Quick Start！](https://www.paddlepaddle.org.cn/inference/master/guides/quick_start/index_quick_start.html)
- [Paddle Inference C++ Api使用](https://www.paddlepaddle.org.cn/inference/master/api_reference/cxx_api_doc/cxx_api_index.html)
- [Paddle Inference Python Api使用](https://www.paddlepaddle.org.cn/inference/master/api_reference/python_api_doc/python_api_index.html)
