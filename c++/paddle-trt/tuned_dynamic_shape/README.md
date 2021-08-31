## 使用Paddle-TRT TunedDynamicShape能力

该文档为使用Paddle-TRT TunedDynamicShape的实践demo。如果您刚接触Paddle-TRT，推荐先访问[这里](https://paddle-inference.readthedocs.io/en/latest/optimize/paddle_trt.html)对Paddle-TRT有个初步认识。

### 获取paddle_inference预测库

下载paddle_inference预测库并解压存储到`Paddle-Inference-Demo/c++/lib`目录，lib目录结构如下所示

```
Paddle-Inference-Demo/c++/lib/
├── CMakeLists.txt
└── paddle_inference
    ├── CMakeCache.txt
    ├── paddle
    │   ├── include                                    C++ 预测库头文件目录
    │   │   ├── crypto
    │   │   ├── internal
    │   │   ├── paddle_analysis_config.h
    │   │   ├── paddle_api.h
    │   │   ├── paddle_infer_declare.h
    │   │   ├── paddle_inference_api.h                 C++ 预测库头文件
    │   │   ├── paddle_mkldnn_quantizer_config.h
    │   │   └── paddle_pass_builder.h
    │   └── lib
    │       ├── libpaddle_inference.a                  C++ 静态预测库文件
    │       └── libpaddle_inference.so                 C++ 动态态预测库文件
    ├── third_party
    │   ├── install                                    第三方链接库和头文件
    │   │   ├── cryptopp
    │   │   ├── gflags
    │   │   ├── glog
    │   │   ├── mkldnn
    │   │   ├── mklml
    │   │   ├── protobuf
    │   │   └── xxhash
    │   └── threadpool
    │       └── ThreadPool.h
    └── version.txt
```

本目录下，

- `clas.cc` 为使用Paddle-TRT TunedDynamicShape针对PaddleClas产出模型的测试程序源文件（程序中的输入为固定值，如果您有opencv或其他方式进行数据读取的需求，需要对程序进行一定的修改）。
- `detect.cc` 为使用Paddle-TRT TunedDynamicShape针对PaddleDetection产出模型的测试程序源文件（程序中的输入为固定值，如果您有opencv或其他方式进行数据读取的需求，需要对程序进行一定的修改）。
- `ocr_cls.cc` 为使用Paddle-TRT TunedDynamicShape针对PaddleOCR产出cls模型的测试程序源文件（程序中的输入为固定值，如果您有opencv或其他方式进行数据读取的需求，需要对程序进行一定的修改）。
- `ocr_det.cc` 为使用Paddle-TRT TunedDynamicShape针对PaddleOCR产出det模型的测试程序源文件（程序中的输入为固定值，如果您有opencv或其他方式进行数据读取的需求，需要对程序进行一定的修改）。
- `ocr_rec.cc` 为使用Paddle-TRT TunedDynamicShape针对PaddleOCR产出rec模型的测试程序源文件（程序中的输入为固定值，如果您有opencv或其他方式进行数据读取的需求，需要对程序进行一定的修改）。
- `ernie.cc`
- `bert.cc`
- `compile.sh` 包含了第三方库、预编译库的信息配置。

### PaddleClas模型示例

[PaddleClas](https://github.com/PaddlePaddle/PaddleClas)是飞桨的图像识别套件，是为工业界和学术界所准备的一个图像识别任务的工具集，助力使用者训练出更好的视觉模型和应用落地。

#### 获取PaddleClas模型

您可以参考PaddleClas的[文档](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.2/docs/zh_CN/inference.md)来获取其提供的预测格式模型，此处，只列出部分模型的获取及转换方式。

```
git clone https://github.com/PaddlePaddle/PaddleClas.git
cd PaddleClas
git checkout release/2.2
mkdir pretrained
mkdir inference_model

# ResNet50_vd
wget --no-proxy -P pretrained/ https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/ResNet50_vd_pretrained.pdparams
python tools/export_model.py -c ppcls/configs/ImageNet/ResNet/ResNet50_vd.yaml -o Global.pretrained_model=./pretrained/ResNet50_vd_pretrained -o Global.save_inference_dir=./inference_model/ResNet50_vd

# GhostNet_x1_0
wget --no-proxy -P pretrained/ https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/GhostNet_x1_0_pretrained.pdparams
python tools/export_model.py -c ppcls/configs/ImageNet/GhostNet/GhostNet_x1_0.yaml -o Global.pretrained_model=./pretrained/GhostNet_x1_0_pretrained -o Global.save_inference_dir=./inference_model/GhostNet_x1_0

# DenseNet121
wget --no-proxy -P pretrained/ https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DenseNet121_pretrained.pdparams
python tools/export_model.py -c ppcls/configs/ImageNet/DenseNet/DenseNet121.yaml -o Global.pretrained_model=./pretrained/DenseNet121_pretrained -o Global.save_inference_dir=./inference_model/DenseNet121

# HRNet_W32_C
wget --no-proxy -P pretrained/ https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/HRNet_W32_C_pretrained.pdparams
python tools/export_model.py -c ppcls/configs/ImageNet/HRNet/HRNet_W32_C.yaml -o Global.pretrained_model=./pretrained/HRNet_W32_C_pretrained -o Global.save_inference_dir=./inference_model/HRNet_W32_C

# InceptionV4
wget --no-proxy -P pretrained/ https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/InceptionV4_pretrained.pdparams
python tools/export_model.py -c ppcls/configs/ImageNet/Inception/InceptionV4.yaml -o Global.pretrained_model=./pretrained/InceptionV4_pretrained -o Global.save_inference_dir=./inference_model/InceptionV4

# ViT_base_patch16_224
wget --no-proxy -P pretrained/ https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/EfficientNetB3_pretrained.pdparams
python tools/export_model.py -c ppcls/configs/ImageNet/VisionTransformer/ViT_base_patch16_224.yaml -o Global.pretrained_model=./pretrained/ViT_base_patch16_224_pretrained -o Global.save_inference_dir=./inference_model/ViT_base_patch16_224

# DeiT_base_patch16_224
wget --no-proxy -P pretrained/ https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DeiT_base_patch16_224_pretrained.pdparams
python tools/export_model.py -c ppcls/configs/ImageNet/DeiT/DeiT_base_patch16_224.yaml -o Global.pretrained_model=./pretrained/DeiT_base_patch16_224_pretrained -o Global.save_inference_dir=./inference_model/DeiT_base_patch16_224
```

预测格式模型均存储在`inference_model`目录下，为方便后续测试，将`inference_model`软链到`Paddle-Inference-Demo/c++/paddle-trt/tuned_dynamic_shape`目录下。


#### TunedDynamicShape测试

首先，您需要编译单测（请注意您可能需要对compile.sh中的配置信息进行修改），在build目录下产出可执行文件`clas`

```
sh compile.sh clas
```

**1、首先您需要针对业务数据进行离线tune，来获取计算图中所有中间tensor的shape范围，并将其存储在config中配置的shape_range_info.pbtxt文件中**

```
./build/clas --model_file inference_model/GhostNet_x1_0/inference.pdmodel --params_file inference_model/GhostNet_x1_0/inference.pdiparams --hs="224:448" --ws="224:448" --tune
```

**2、有了离线tune得到的shape范围信息后，您可以使用该文件自动对所有的trt子图设置其输入的shape范围。

```
./build/clas --model_file inference_model/GhostNet_x1_0/inference.pdmodel --params_file inference_model/GhostNet_x1_0/inference.pdiparams --hs="224:448" --ws="224:448" --no_seen_hs="112:672" --no_seen_ws="112:672" --tuned_dynamic_shape
```

完整测试case：

```shell
# 注意以下CV类的测试case，不是真正的变长模型，对输入shape是有要求的，无法支持任意的输入shape

# ResNet50_Vd
# 离线tune测试
./build/clas --model_file inference_model/ResNet50_vd/inference.pdmodel  --params_file inference_model/ResNet50_vd/inference.pdiparams --hs="224:448" --ws="224:448" --tune
# 动态shape及序列化测试
./build/clas --model_file inference_model/ResNet50_vd/inference.pdmodel  --params_file inference_model/ResNet50_vd/inference.pdiparams --hs="224:448" --ws="224:448" --no_seen_hs="112:672" --no_seen_ws="112:672" --tuned_dynamic_shape --serialize
# 动态shape及反序列化测试
./build/clas --model_file inference_model/ResNet50_vd/inference.pdmodel  --params_file inference_model/ResNet50_vd/inference.pdiparams --hs="224:448" --ws="224:448" --no_seen_hs="112:672" --no_seen_ws="112:672" --tuned_dynamic_shape --serialize

# GhostNet_x1_0
# 离线tune测试
./build/clas --model_file inference_model/GhostNet_x1_0/inference.pdmodel --params_file inference_model/GhostNet_x1_0/inference.pdiparams --hs="224:448" --ws="224:448" --tune
# 动态shape及序列化测试
./build/clas --model_file inference_model/GhostNet_x1_0/inference.pdmodel --params_file inference_model/GhostNet_x1_0/inference.pdiparams --hs="224:448" --ws="224:448" --no_seen_hs="112:672" --no_seen_ws="112:672" --tuned_dynamic_shape --serialize
# 动态shape及反序列化测试
./build/clas --model_file inference_model/GhostNet_x1_0/inference.pdmodel --params_file inference_model/GhostNet_x1_0/inference.pdiparams --hs="224:448" --ws="224:448" --no_seen_hs="112:672" --no_seen_ws="112:672" --tuned_dynamic_shape --serialize

# DenseNet121
# 离线tune测试
./build/clas --model_file inference_model/DenseNet121/inference.pdmodel --params_file inference_model/DenseNet121/inference.pdiparams --hs="224:448" --ws="224:448" --tune
# 动态shape及序列化测试
./build/clas --model_file inference_model/DenseNet121/inference.pdmodel --params_file inference_model/DenseNet121/inference.pdiparams --hs="224:448" --ws="224:448" --no_seen_hs="112:672" --no_seen_ws="112:672" --tuned_dynamic_shape --serialize
# 动态shape及反序列化测试
./build/clas --model_file inference_model/DenseNet121/inference.pdmodel --params_file inference_model/DenseNet121/inference.pdiparams --hs="224:448" --ws="224:448" --no_seen_hs="112:672" --no_seen_ws="112:672" --tuned_dynamic_shape --serialize

# HRNet_W32_C
# 离线tune测试
./build/clas --model_file inference_model/HRNet_W32_C/inference.pdmodel --params_file inference_model/HRNet_W32_C/inference.pdiparams --hs="224" --ws="224" --tune
# 动态shape及序列化测试
./build/clas --model_file inference_model/HRNet_W32_C/inference.pdmodel --params_file inference_model/HRNet_W32_C/inference.pdiparams --hs="224" --ws="224" --no_seen_hs="448" --no_seen_ws="448" --tuned_dynamic_shape --serialize
# 动态shape及反序列化测试
./build/clas --model_file inference_model/HRNet_W32_C/inference.pdmodel --params_file inference_model/HRNet_W32_C/inference.pdiparams --hs="224" --ws="224" --no_seen_hs="448" --no_seen_ws="448" --tuned_dynamic_shape --serialize

# InceptionV4
# 离线tune测试
./build/clas --model_file inference_model/InceptionV4/inference.pdmodel --params_file inference_model/InceptionV4/inference.pdiparams --hs="224" --ws="224" --tune
# 动态shape及序列化测试
./build/clas --model_file inference_model/InceptionV4/inference.pdmodel --params_file inference_model/InceptionV4/inference.pdiparams --hs="224" --ws="224" --no_seen_hs="448" --no_seen_ws="448" --tuned_dynamic_shape --serialize
# 动态shape及反序列化测试
./build/clas --model_file inference_model/InceptionV4/inference.pdmodel --params_file inference_model/InceptionV4/inference.pdiparams --hs="224" --ws="224" --no_seen_hs="448" --no_seen_ws="448" --tuned_dynamic_shape --serialize

# ViT_base_patch16_224
# 离线tune测试
./build/clas --model_file inference_model/ViT_base_patch16_224/inference.pdmodel --params_file inference_model/ViT_base_patch16_224/inference.pdiparams --hs="224" --ws="224" --tune
# 动态shape及序列化测试（注意：目前需要config加配置：config->Exp_DisableTensorRtOPs({"elementwise_add"});）
./build/clas --model_file inference_model/ViT_base_patch16_224/inference.pdmodel --params_file inference_model/ViT_base_patch16_224/inference.pdiparams --hs="224" --ws="224"--no_seen_hs="224" --no_seen_ws="224" --tuned_dynamic_shape --serialize
# 动态shape及反序列化测试（注意：目前需要config加配置：config->Exp_DisableTensorRtOPs({"elementwise_add"});）
./build/clas --model_file inference_model/ViT_base_patch16_224/inference.pdmodel --params_file inference_model/ViT_base_patch16_224/inference.pdiparams --hs="224" --ws="224"--no_seen_hs="224" --no_seen_ws="224" --tuned_dynamic_shape --serialize

# DeiT_base_patch16_224
# 离线tune测试
./build/clas --model_file inference_model/DeiT_base_patch16_224/inference.pdmodel --params_file inference_model/DeiT_base_patch16_224/inference.pdiparams --hs="224" --ws="224" --tune
# 动态shape及序列化测试（注意：目前需要config加配置：config->Exp_DisableTensorRtOPs({"elementwise_add"});）
./build/clas --model_file inference_model/DeiT_base_patch16_224/inference.pdmodel --params_file inference_model/DeiT_base_patch16_224/inference.pdiparams --hs="224" --ws="224"--no_seen_hs="224" --no_seen_ws="224" --tuned_dynamic_shape --serialize
# 动态shape及反序列化测试（注意：目前需要config加配置：config->Exp_DisableTensorRtOPs({"elementwise_add"});）
./build/clas --model_file inference_model/DeiT_base_patch16_224/inference.pdmodel --params_file inference_model/DeiT_base_patch16_224/inference.pdiparams --hs="224" --ws="224"--no_seen_hs="224" --no_seen_ws="224" --tuned_dynamic_shape --serialize
```

### PaddleDetection模型示例

[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)是飞桨的目标检测开发套件，模块化地实现了多种主流目标检测算法，提供了丰富的数据增强策略、网络模块组件（如骨干网络）、损失函数等，并集成了模型压缩和跨平台高性能部署能力。

#### 获取PaddleDetection模型

您可以参考PaddleDetection的[模型导出教程](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.1/deploy/EXPORT_MODEL.md)来获取其提供的预测格式模型，此处，只列出部分模型的获取及转换方式。

```
git clone https://github.com/PaddlePaddle/PaddleDetection.git
cd PaddleDetection
git checkout release/2.1
mkdir pretrained
mkdir inference_model

# faster_rcnn_r50_fpn_1x_coco
wget --no-proxy -P pretrained/ https://paddledet.bj.bcebos.com/models/faster_rcnn_r50_fpn_1x_coco.pdparams
python tools/export_model.py -c configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.yml --output_dir=./inference_model/ -o weights=pretrained/faster_rcnn_r50_fpn_1x_coco.pdparams

# mask_rcnn_r50_vd_fpn_2x_coco
wget --no-proxy -P pretrained/ https://paddledet.bj.bcebos.com/models/mask_rcnn_r50_vd_fpn_2x_coco.pdparams
python tools/export_model.py -c configs/mask_rcnn/mask_rcnn_r50_vd_fpn_2x_coco.yml --output_dir=./inference_model/ -o weights=pretrained/mask_rcnn_r50_vd_fpn_2x_coco.pdparams

# yolov3_darknet53_270e_coco
wget --no-proxy -P pretrained/ https://paddledet.bj.bcebos.com/models/yolov3_darknet53_270e_coco.pdparams
python tools/export_model.py -c configs/yolov3/yolov3_darknet53_270e_coco.yml --output_dir=./inference_model/ -o weights=pretrained/yolov3_darknet53_270e_coco.pdparams

# ssd_mobilenet_v1_300_120e_voc
wget --no-proxy -P pretrained/ https://paddledet.bj.bcebos.com/models/ssd_mobilenet_v1_300_120e_voc.pdparams
python tools/export_model.py -c configs/ssd/ssd_mobilenet_v1_300_120e_voc.yml --output_dir=./inference_model/ -o weights=pretrained/ssd_mobilenet_v1_300_120e_voc.pdparams
```

预测格式模型均存储在`inference_model`目录下，为方便测试，将`inference_model`软链到`Paddle-Inference-Demo/c++/paddle-trt/tuned_dynamic_shape`目录下（您可能需要删除之前建立的软链）。

#### TunedDynamicShape测试

首先，您需要编译单测（请注意您可能需要对compile.sh中的配置信息进行修改），在build目录下产出可执行文件`detect`

```
sh compile.sh detect
```

**1、首先您需要针对业务数据进行离线tune，来获取计算图中所有中间tensor的shape范围，并将其存储在config中配置的shape_range_info.pbtxt文件中**

```
./build/detect --model_file inference_model/yolov3_darknet53_270e_coco/model.pdmodel --params_file inference_model/yolov3_darknet53_270e_coco/model.pdiparams --hs="608" --ws="608" --tune
```

**2、有了离线tune得到的shape范围信息后，您可以使用该文件自动对所有的trt子图设置其输入的shape范围。

```
./build/detect --model_file inference_model/yolov3_darknet53_270e_coco/model.pdmodel --params_file inference_model/yolov3_darknet53_270e_coco/model.pdiparams --hs="608" --ws="608" --no_seen_hs="416" --no_seen_ws="416" --tuned_dynamic_shape
```

完整测试case：
```shell
# 注意以下CV类的测试case，不是真正的变长模型，对输入shape是有要求的，无法支持任意的输入shape

# yolov3_darknet53_270e_coco
# 离线tune测试
./build/detect --model_file inference_model/yolov3_darknet53_270e_coco/model.pdmodel --params_file inference_model/yolov3_darknet53_270e_coco/model.pdiparams --hs="608" --ws="608" --tune
# 动态shape及序列化测试
./build/detect --model_file inference_model/yolov3_darknet53_270e_coco/model.pdmodel --params_file inference_model/yolov3_darknet53_270e_coco/model.pdiparams --hs="608" --ws="608" --no_seen_hs="416" --no_seen_ws="416" --tuned_dynamic_shape --serialize
# 动态shape及反序列化测试
./build/detect --model_file inference_model/yolov3_darknet53_270e_coco/model.pdmodel --params_file inference_model/yolov3_darknet53_270e_coco/model.pdiparams --hs="608" --ws="608" --no_seen_hs="416" --no_seen_ws="416" --tuned_dynamic_shape --serialize


# ssd_mobilenet_v1_300_120e_voc
# 离线tune测试
./build/detect --model_file inference_model/ssd_mobilenet_v1_300_120e_voc//model.pdmodel --params_file inference_model/ssd_mobilenet_v1_300_120e_voc//model.pdiparams --hs="300" --ws="300" --tune
# 动态shape及序列化测试（注意：目前需要config加配置：config->Exp_DisableTensorRtOPs({"elementwise_add"});）
./build/detect --model_file inference_model/ssd_mobilenet_v1_300_120e_voc//model.pdmodel --params_file inference_model/ssd_mobilenet_v1_300_120e_voc//model.pdiparams --hs="300" --ws="300" --no_seen_hs="416" --no_seen_ws="416" --tuned_dynamic_shape --serialize
# 动态shape及反序列化测试（注意：目前需要config加配置：config->Exp_DisableTensorRtOPs({"elementwise_add"});）
./build/detect --model_file inference_model/ssd_mobilenet_v1_300_120e_voc//model.pdmodel --params_file inference_model/ssd_mobilenet_v1_300_120e_voc//model.pdiparams --hs="300" --ws="300" --no_seen_hs="416" --no_seen_ws="416" --tuned_dynamic_shape --serialize

# faster_rcnn_r50_fpn_1x_coco
# 离线tune测试
./build/detect --model_file inference_model/faster_rcnn_r50_fpn_1x_coco/model.pdmodel --params_file inference_model/faster_rcnn_r50_fpn_1x_coco/model.pdiparams --hs="608" --ws="608" --tune
# 动态shape及序列化测试（注意：目前需要config加配置：config->Exp_DisableTensorRtOPs({"roi_align", "elementwise_add"});）
./build/detect --model_file inference_model/faster_rcnn_r50_fpn_1x_coco/model.pdmodel --params_file inference_model/faster_rcnn_r50_fpn_1x_coco/model.pdiparams --hs="608" --ws="608" --no_seen_hs="416" --no_seen_ws="416" --tuned_dynamic_shape --serialize
# 动态shape及反序列化测试（注意：目前需要config加配置：config->Exp_DisableTensorRtOPs({"roi_align", "elementwise_add"});）
./build/detect --model_file inference_model/faster_rcnn_r50_fpn_1x_coco/model.pdmodel --params_file inference_model/faster_rcnn_r50_fpn_1x_coco/model.pdiparams --hs="608" --ws="608" --no_seen_hs="416" --no_seen_ws="416" --tuned_dynamic_shape --serialize

# mask_rcnn_r50_vd_fpn_2x_coco
# 离线tune测试
./build/detect --model_file inference_model/mask_rcnn_r50_vd_fpn_2x_coco/model.pdmodel --params_file inference_model/mask_rcnn_r50_vd_fpn_2x_coco/model.pdiparams --hs="608" --ws="608" --tune
# 动态shape及序列化测试（注意：目前需要config加配置：config->Exp_DisableTensorRtOPs({"roi_align", "elementwise_add"});）
./build/detect --model_file inference_model/mask_rcnn_r50_vd_fpn_2x_coco/model.pdmodel --params_file inference_model/mask_rcnn_r50_vd_fpn_2x_coco/model.pdiparams --hs="608" --ws="608" --no_seen_hs="416" --no_seen_ws="416" --tuned_dynamic_shape --serialize
# 动态shape及反序列化测试（注意：目前需要config加配置：config->Exp_DisableTensorRtOPs({"roi_align", "elementwise_add"});）
./build/detect --model_file inference_model/mask_rcnn_r50_vd_fpn_2x_coco/model.pdmodel --params_file inference_model/mask_rcnn_r50_vd_fpn_2x_coco/model.pdiparams --hs="608" --ws="608" --no_seen_hs="416" --no_seen_ws="416" --tuned_dynamic_shape --serialize
```

### PaddleOCR模型测试

[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)是一套丰富、领先、且实用的OCR工具库，助力使用者训练出更好的模型，并应用落地。

#### 获取PaddleOCR模型

您可以从PaddleOCR的[模型列表](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.2/README_ch.md)中直接获取其提供的预测格式模型。

```
mkdir ocr_inference_model && cd ocr_inference_model

wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_infer.tar
tar xf ch_ppocr_mobile_v2.0_det_infer.tar

wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar
tar xf ch_ppocr_mobile_v2.0_cls_infer.tar

wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_infer.tar
tar xf ch_ppocr_mobile_v2.0_rec_infer.tar

wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_det_infer.tar
tar xf ch_ppocr_server_v2.0_det_infer.tar

wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_rec_infer.tar
tar xf ch_ppocr_server_v2.0_rec_infer.tar
```

#### TunedDynamicShape测试

首先，您需要编译单测（请注意您可能需要对compile.sh中的配置信息进行修改），在build目录下产出可执行文件`ocr`

```shell
# ocr det模型测试
sh compile.sh ocr_det

# 离线tune测试
./build/ocr_det --model_file ocr_inference_model/ch_ppocr_server_v2.0_det_infer/inference.pdmodel --params_file ocr_inference_model/ch_ppocr_server_v2.0_det_infer/inference.pdiparams --hs="640" --ws="640" --tune
# 动态shape及序列化测试
./build/ocr_det --model_file ocr_inference_model/ch_ppocr_server_v2.0_det_infer/inference.pdmodel --params_file ocr_inference_model/ch_ppocr_server_v2.0_det_infer/inference.pdiparams --hs="640" --ws="640" --no_seen_hs="320" --no_seen_ws="320" --tuned_dynamic_shape --serialize
# 动态shape及反序列化测试
./build/ocr_det --model_file ocr_inference_model/ch_ppocr_server_v2.0_det_infer/inference.pdmodel --params_file ocr_inference_model/ch_ppocr_server_v2.0_det_infer/inference.pdiparams --hs="640" --ws="640" --no_seen_hs="320" --no_seen_ws="320" --tuned_dynamic_shape --serialize

# ocr cls模型测试
sh compile.sh ocr_cls

# 离线tune测试
./build/ocr_cls --model_file ocr_inference_model/ch_ppocr_mobile_v2.0_cls_infer/inference.pdmodel --params_file ocr_inference_model/ch_ppocr_mobile_v2.0_cls_infer/inference.pdiparams --hs="640" --ws="640" --tune
# 动态shape及序列化测试
./build/ocr_cls --model_file ocr_inference_model/ch_ppocr_mobile_v2.0_cls_infer/inference.pdmodel --params_file ocr_inference_model/ch_ppocr_mobile_v2.0_cls_infer/inference.pdiparams --hs="640" --ws="640" --no_seen_hs="320" --no_seen_ws="320" --tuned_dynamic_shape --serialize
# 动态shape及反序列化测试
./build/ocr_cls --model_file ocr_inference_model/ch_ppocr_mobile_v2.0_cls_infer/inference.pdmodel --params_file ocr_inference_model/ch_ppocr_mobile_v2.0_cls_infer/inference.pdiparams --hs="640" --ws="640" --no_seen_hs="320" --no_seen_ws="320" --tuned_dynamic_shape --serialize


# ocr rec模型测试
sh compile.sh ocr_rec

# 离线tune测试
./build/ocr_rec --model_file ocr_inference_model/ch_ppocr_server_v2.0_rec_infer/inference.pdmodel --params_file ocr_inference_model/ch_ppocr_server_v2.0_rec_infer/inference.pdiparams --hs="32" --ws="32" --tune
# 动态shape及序列化测试
./build/ocr_rec --model_file ocr_inference_model/ch_ppocr_server_v2.0_rec_infer/inference.pdmodel --params_file ocr_inference_model/ch_ppocr_server_v2.0_rec_infer/inference.pdiparams --hs="32" --ws="32" --no_seen_hs="32" --no_seen_ws="320" --tuned_dynamic_shape --serialize
# 动态shape及反序列化测试
./build/ocr_rec --model_file ocr_inference_model/ch_ppocr_server_v2.0_rec_infer/inference.pdmodel --params_file ocr_inference_model/ch_ppocr_server_v2.0_rec_infer/inference.pdiparams --hs="32" --ws="32" --no_seen_hs="32" --no_seen_ws="320" --tuned_dynamic_shape --serialize
```



### PaddleNLP模型测试

[PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP)是飞桨生态的文本领域核心库，具备易用的文本领域API，多场景的应用示例、和高性能分布式训练三大特点，旨在提升开发者文本领域的开发效率，并提供基于飞桨2.0核心框架的NLP任务最佳实践。

#### 获取PaddleNLP模型

详情请参考[文档](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/text_classification/pretrained_models)

测试
```shell
mkdir nlp_inference_model && cd nlp_inference_model
wget -P nlp_inference_model http://paddle-inference-dist.bj.bcebos.com/tensorrt_test/ernie_model_4.tar.gz
tar xzf ernie_model_4.tar.gz
cd -

sh compile.sh ernie

# 离线tune测试
./build/ernie --model_dir nlp_inference_model/ernie_model_4 --tune
# 动态shape及序列化测试
./build/ernie --model_dir nlp_inference_model/ernie_model_4 --tuned_dynamic_shape --serialize
# 动态shape及反序列化测试
./build/ernie --model_dir nlp_inference_model/ernie_model_4 --tuned_dynamic_shape --serialize
```

#### ernie_varlen 变长的支持

目前ernie变长暂不支持tuned动态Shape设置，to be continued...

### 更多链接
- [Paddle Inference使用Quick Start！](https://paddle-inference.readthedocs.io/en/latest/introduction/quick_start.html)
- [Paddle Inference C++ Api使用](https://paddle-inference.readthedocs.io/en/latest/user_guides/cxx_api.html)
- [Paddle Inference Python Api使用](https://paddle-inference.readthedocs.io/en/latest/user_guides/inference_python_api.html)

