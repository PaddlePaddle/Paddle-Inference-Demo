# GPU 性能数据

## 测试条件

- 测试模型
	- MobileNetV1
	- MobileNetV2
	- ResNet101
	- mask_rcnn_r50_vd_fpn_1x_coco
	- ssdlite_mobilenet_v1_300_coco
	- yolov3_darknet53_270e_coco
	- deeplabv3p_resnet50
	- bert
	- ViT_base_patch32_384
	- SwinTransformer_base_patch4_window12_384

- 测试机器信息
	- NVIDIA® T4 GPU
	- Intel(R) Xeon(R) Gold 6271C CPU @ 2.60GHz
	- CUDA 11.2.2
	- cuDNN 8.2.1
	- TensorRT 8.0.3.4
- 测试说明
	- 测试 PaddlePaddle 版本：v2.3
	- warmup=10，repeats=1000，统计平均时间，单位为 ms。
	- cpu_math_library_num_threads=1，num_samples=1000。

## 数据

|	model_name	|	precision	|	batch_size	|	avg_latency	|
|-|-|-|-|
|	MobileNetV1	|	fp16	|	1	|	0.4925	|
|	MobileNetV1	|	fp16	|	2	|	0.7485	|
|	MobileNetV1	|	fp16	|	4	|	1.2914	|
|	MobileNetV1	|	fp32	|	1	|	0.8737	|
|	MobileNetV1	|	fp32	|	2	|	1.4106	|
|	MobileNetV1	|	fp32	|	4	|	2.5238	|
|	MobileNetV2	|	fp16	|	1	|	0.5926	|
|	MobileNetV2	|	fp16	|	2	|	0.9131	|
|	MobileNetV2	|	fp16	|	4	|	1.4491	|
|	MobileNetV2	|	fp32	|	1	|	1.1125	|
|	MobileNetV2	|	fp32	|	2	|	1.6682	|
|	MobileNetV2	|	fp32	|	4	|	2.819	|
|	ResNet101	|	fp16	|	1	|	2.1345	|
|	ResNet101	|	fp16	|	2	|	2.9835	|
|	ResNet101	|	fp16	|	4	|	4.9308	|
|	ResNet101	|	fp32	|	1	|	6.3175	|
|	ResNet101	|	fp32	|	2	|	9.251	|
|	ResNet101	|	fp32	|	4	|	16.7459	|
|	SwinTransformer_base_patch4_window12_384	|	fp16	|	1	|	23.0886	|
|	SwinTransformer_base_patch4_window12_384	|	fp16	|	2	|	42.2748	|
|	SwinTransformer_base_patch4_window12_384	|	fp16	|	4	|	87.3252	|
|	SwinTransformer_base_patch4_window12_384	|	fp32	|	1	|	43.5075	|
|	SwinTransformer_base_patch4_window12_384	|	fp32	|	2	|	87.5455	|
|	SwinTransformer_base_patch4_window12_384	|	fp32	|	4	|	173.796	|
|	ViT_base_patch32_384	|	fp16	|	1	|	4.923	|
|	ViT_base_patch32_384	|	fp16	|	2	|	7.5347	|
|	ViT_base_patch32_384	|	fp16	|	4	|	12.899	|
|	ViT_base_patch32_384	|	fp32	|	1	|	10.8246	|
|	ViT_base_patch32_384	|	fp32	|	2	|	18.5213	|
|	ViT_base_patch32_384	|	fp32	|	4	|	34.7381	|
|	deeplabv3p_resnet50	|	fp16	|	1	|	26.1575	|
|	deeplabv3p_resnet50	|	fp16	|	2	|	47.9256	|
|	deeplabv3p_resnet50	|	fp16	|	4	|	95.9487	|
|	deeplabv3p_resnet50	|	fp32	|	1	|	66.8809	|
|	deeplabv3p_resnet50	|	fp32	|	2	|	133.6688	|
|	deeplabv3p_resnet50	|	fp32	|	4	|	266.9613	|
|	mask_rcnn_r50_vd_fpn_1x_coco	|	fp16	|	1	|	40.6577	|
|	mask_rcnn_r50_vd_fpn_1x_coco	|	fp32	|	1	|	101.93	|
|	yolov3_darknet53_270e_coco_upload	|	fp16	|	1	|	20.6326	|
|	yolov3_darknet53_270e_coco_upload	|	fp16	|	2	|	41.5202	|
|	yolov3_darknet53_270e_coco_upload	|	fp16	|	4	|	80.3059	|
|	yolov3_darknet53_270e_coco_upload	|	fp32	|	1	|	44.1216	|
|	yolov3_darknet53_270e_coco_upload	|	fp32	|	2	|	85.4666	|
|	yolov3_darknet53_270e_coco_upload	|	fp32	|	4	|	183.9448	|