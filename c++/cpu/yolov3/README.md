# 运行 YOLOv3 图像检测样例

YOLOv3 样例展示了多输入模型在 CPU 下的推理过程。运行步骤如下：

## 一：获取 Paddle Inference 预测库

- [官网下载](https://www.paddlepaddle.org.cn/documentation/docs/zh/advanced_guide/inference_deployment/inference/build_and_install_lib_cn.html)
- 自行编译获取

将获取到的 Paddle Inference 预测库软链接或者重命名为 `paddle_inference`，并置于 `Paddle-Inference-Demo/c++/lib` 目录下。

## 二：获取 YOLOv3 模型

点击[链接](https://paddle-inference-dist.bj.bcebos.com/Paddle-Inference-Demo/yolov3_r50vd_dcn_270e_coco.tgz)下载模型，如果你想获取更多的**模型训练信息**，请访问[这里](https://github.com/PaddlePaddle/PaddleDetection)。

或者使用如下命令：
```
wget https://paddle-inference-dist.bj.bcebos.com/Paddle-Inference-Demo/yolov3_r50vd_dcn_270e_coco.tgz
tar xzf yolov3_r50vd_dcn_270e_coco.tgz
```

## 三：编译样例
 
- 文件`yolov3_test.cc` 为预测的样例程序（程序中的输入为固定值，如果您有opencv或其他方式进行数据读取的需求，需要对程序进行一定的修改）。    
- 脚本`compile.sh` 包含了第三方库、预编译库的信息配置。
- 脚本`run.sh`为一键运行脚本。

运行 `bash compile.sh` 编译样例。

## 四：运行样例

### 使用 oneDNN 运行样例

```shell
./build/yolov3_test --model_file yolov3_r50vd_dcn_270e_coco/model.pdmodel --params_file yolov3_r50vd_dcn_270e_coco/model.pdiparams
```

### 使用 OnnxRuntime 运行样例
```shell
./build/yolov3_test --model_file yolov3_r50vd_dcn_270e_coco/model.pdmodel --params_file yolov3_r50vd_dcn_270e_coco/model.pdiparams --use_ort=1
```

运行结束后，程序会将模型结果打印到屏幕，说明运行成功。

## 更多链接
- [Paddle Inference使用Quick Start！](https://www.paddlepaddle.org.cn/inference/master/guides/quick_start/index_quick_start.html)
- [Paddle Inference C++ Api使用](https://www.paddlepaddle.org.cn/inference/master/api_reference/cxx_api_doc/cxx_api_index.html)
- [Paddle Inference Python Api使用](https://www.paddlepaddle.org.cn/inference/master/api_reference/python_api_doc/python_api_index.html)
