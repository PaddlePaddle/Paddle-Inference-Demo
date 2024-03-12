# 运行自定义Pass调用自定义算子示例

## 一：简介
基于`ResNet50`模型，我们实现一个自定义`Pass`将模型中的`Paddle`原生`pd_op.relu`算子替换成自定义`custom_op.custom_relu`算子。

## 二：样例运行

文件 `custom_relu_op.cc`、`custom_relu_op.cu` 为自定义算子源文件，自定义算子编写方式请参考[飞桨官网文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/07_new_op/new_custom_op.html)。`custom_relu_pass.cc`为自定义Pass源文件。

文件`infer_resnet.py` 为预测的样例程序。  

执行命令：

```
pip install paddlepaddle-gpu==2.6.2
python setup.py install
bash run.sh
```

作为调试，如果你开启`config.switch_ir_debug()`，会看到跑完Pass前后的日志，可以进一步观察到模型中的`pd_op.relu`被`custom_op.custom_relu`替换了。
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
