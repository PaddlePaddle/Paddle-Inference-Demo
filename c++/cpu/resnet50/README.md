# Linux 运行 ResNet50 图像分类样例

ResNet50 样例展示了单输入模型在 CPU 下使用 oneDNN 和 OnnxRuntime 的推理过程。

Windows 运行请看文档: [Windows 运行 ResNet50 图像分类样例](Windows.md)

Linux运行步骤如下：

## 一：获取 Paddle Inference 预测库

- [官网下载](https://www.paddlepaddle.org.cn/documentation/docs/zh/advanced_guide/inference_deployment/inference/build_and_install_lib_cn.html)
- 自行编译获取

将获取到的 Paddle Inference 预测库软链接或者重命名为 `paddle_inference`，并置于 `Paddle-Inference-Demo/c++/lib` 目录下。

## 二：获取 Resnet50 模型

点击[链接](https://paddle-inference-dist.bj.bcebos.com/Paddle-Inference-Demo/resnet50.tgz)下载模型。如果你想获取更多的**模型训练信息**，请访问[这里](https://github.com/PaddlePaddle/PaddleClas)。

## 三：编译样例
 
- 文件`resnet50_test.cc` 为预测的样例程序（程序中的输入为固定值，如果您有 opencv 或其他方式进行数据读取的需求，需要对程序进行一定的修改）。    
- 脚本`compile.sh` 包含了第三方库、预编译库的信息配置。  
- 脚本`run.sh` 为一键运行脚本。

编译前，需要根据自己的环境修改 `compile.sh` 中的相关代码配置依赖库：

```bash
# 根据预编译库中的version.txt信息判断是否将以下标记打开
WITH_MKL=OFF
WITH_ARM=OFF  # 在飞腾/鲲鹏 CPU 下运行，则修改为ON
WITH_MIPS=OFF # 在龙芯 CPU 下运行，则修改为ON
WITH_SW=OFF   # 在申威 CPU 下运行，则修改为ON
```

运行 `bash compile.sh` 编译样例。

## 四：运行样例

### 使用 oneDNN 运行样例
```shell
./build/resnet50_test --model_file resnet50/inference.pdmodel --params_file resnet50/inference.pdiparams
```

### 使用 OnnxRuntime 运行样例
```shell
./build/resnet50_test --model_file resnet50/inference.pdmodel --params_file resnet50/inference.pdiparams --use_ort=1
```
test
运行结束后，程序会将模型结果打印到屏幕，说明运行成功。

## 更多链接
- [Paddle Inference使用Quick Start！](https://paddle-inference.readthedocs.io/en/latest/introduction/quick_start.html)
- [Paddle Inference C++ Api使用](https://paddle-inference.readthedocs.io/en/latest/api_reference/cxx_api_index.html)
- [Paddle Inference Python Api使用](https://paddle-inference.readthedocs.io/en/latest/api_reference/python_api_index.html)