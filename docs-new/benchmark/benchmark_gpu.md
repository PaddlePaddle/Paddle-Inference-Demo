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

## 数据

|	model_name	|	device	|	precision	|	num_samples	|	batch_size	|	cpu_math_library_num_threads	|	enable_gpu	|	enable_trt	|	avg_latency	|	qps	|
|-|-|-|-|-|-|-|-|-|-|
|	MobileNetV1	|	T4	|	fp16	|	1000	|	1	|	1	|	True	|	True	|	0.4925	|	2030.46 	|
|	MobileNetV1	|	T4	|	fp16	|	1000	|	2	|	1	|	True	|	True	|	0.7485	|	2672.01 	|
|	MobileNetV1	|	T4	|	fp16	|	1000	|	4	|	1	|	True	|	True	|	1.2914	|	3097.41 	|
|	MobileNetV1	|	T4	|	fp32	|	1000	|	1	|	1	|	True	|	False	|	2.223	|	449.84 	|
|	MobileNetV1	|	T4	|	fp32	|	1000	|	1	|	1	|	True	|	True	|	0.8737	|	1144.56 	|
|	MobileNetV1	|	T4	|	fp32	|	1000	|	2	|	1	|	True	|	False	|	3.5571	|	562.26 	|
|	MobileNetV1	|	T4	|	fp32	|	1000	|	2	|	1	|	True	|	True	|	1.4106	|	1417.84 	|
|	MobileNetV1	|	T4	|	fp32	|	1000	|	4	|	1	|	True	|	False	|	4.0367	|	990.91 	|
|	MobileNetV1	|	T4	|	fp32	|	1000	|	4	|	1	|	True	|	True	|	2.5238	|	1584.91 	|
|	MobileNetV2	|	T4	|	fp16	|	1000	|	1	|	1	|	True	|	True	|	0.5926	|	1687.48 	|
|	MobileNetV2	|	T4	|	fp16	|	1000	|	2	|	1	|	True	|	True	|	0.9131	|	2190.34 	|
|	MobileNetV2	|	T4	|	fp16	|	1000	|	4	|	1	|	True	|	True	|	1.4491	|	2760.33 	|
|	MobileNetV2	|	T4	|	fp32	|	1000	|	1	|	1	|	True	|	False	|	3.7645	|	265.64 	|
|	MobileNetV2	|	T4	|	fp32	|	1000	|	1	|	1	|	True	|	True	|	1.1125	|	898.88 	|
|	MobileNetV2	|	T4	|	fp32	|	1000	|	2	|	1	|	True	|	False	|	4.5273	|	441.76 	|
|	MobileNetV2	|	T4	|	fp32	|	1000	|	2	|	1	|	True	|	True	|	1.6682	|	1198.90 	|
|	MobileNetV2	|	T4	|	fp32	|	1000	|	4	|	1	|	True	|	False	|	6.2238	|	642.69 	|
|	MobileNetV2	|	T4	|	fp32	|	1000	|	4	|	1	|	True	|	True	|	2.819	|	1418.94 	|
|	ResNet101	|	T4	|	fp16	|	1000	|	1	|	1	|	True	|	True	|	2.1345	|	468.49 	|
|	ResNet101	|	T4	|	fp16	|	1000	|	2	|	1	|	True	|	True	|	2.9835	|	670.35 	|
|	ResNet101	|	T4	|	fp16	|	1000	|	4	|	1	|	True	|	True	|	4.9308	|	811.23 	|
|	ResNet101	|	T4	|	fp32	|	1000	|	1	|	1	|	True	|	False	|	15.0993	|	66.23 	|
|	ResNet101	|	T4	|	fp32	|	1000	|	1	|	1	|	True	|	True	|	6.3175	|	158.29 	|
|	ResNet101	|	T4	|	fp32	|	1000	|	2	|	1	|	True	|	False	|	17.5788	|	113.77 	|
|	ResNet101	|	T4	|	fp32	|	1000	|	2	|	1	|	True	|	True	|	9.251	|	216.19 	|
|	ResNet101	|	T4	|	fp32	|	1000	|	4	|	1	|	True	|	False	|	20.6774	|	193.45 	|
|	ResNet101	|	T4	|	fp32	|	1000	|	4	|	1	|	True	|	True	|	16.7459	|	238.86 	|
|	SwinTransformer_base_patch4_window12_384	|	T4	|	fp16	|	1000	|	1	|	1	|	True	|	True	|	23.0886	|	43.31 	|
|	SwinTransformer_base_patch4_window12_384	|	T4	|	fp16	|	1000	|	2	|	1	|	True	|	True	|	42.2748	|	47.31 	|
|	SwinTransformer_base_patch4_window12_384	|	T4	|	fp16	|	1000	|	4	|	1	|	True	|	True	|	87.3252	|	45.81 	|
|	SwinTransformer_base_patch4_window12_384	|	T4	|	fp32	|	1000	|	1	|	1	|	True	|	False	|	42.5669	|	23.49 	|
|	SwinTransformer_base_patch4_window12_384	|	T4	|	fp32	|	1000	|	1	|	1	|	True	|	True	|	43.5075	|	22.98 	|
|	SwinTransformer_base_patch4_window12_384	|	T4	|	fp32	|	1000	|	2	|	1	|	True	|	False	|	84.5615	|	23.65 	|
|	SwinTransformer_base_patch4_window12_384	|	T4	|	fp32	|	1000	|	2	|	1	|	True	|	True	|	87.5455	|	22.85 	|
|	SwinTransformer_base_patch4_window12_384	|	T4	|	fp32	|	1000	|	4	|	1	|	True	|	False	|	169.1416	|	23.65 	|
|	SwinTransformer_base_patch4_window12_384	|	T4	|	fp32	|	1000	|	4	|	1	|	True	|	True	|	173.796	|	23.02 	|
|	ViT_base_patch32_384	|	T4	|	fp16	|	1000	|	1	|	1	|	True	|	True	|	4.923	|	203.13 	|
|	ViT_base_patch32_384	|	T4	|	fp16	|	1000	|	2	|	1	|	True	|	True	|	7.5347	|	265.44 	|
|	ViT_base_patch32_384	|	T4	|	fp16	|	1000	|	4	|	1	|	True	|	True	|	12.899	|	310.10 	|
|	ViT_base_patch32_384	|	T4	|	fp32	|	1000	|	1	|	1	|	True	|	False	|	10.2415	|	97.64 	|
|	ViT_base_patch32_384	|	T4	|	fp32	|	1000	|	1	|	1	|	True	|	True	|	10.8246	|	92.38 	|
|	ViT_base_patch32_384	|	T4	|	fp32	|	1000	|	2	|	1	|	True	|	False	|	17.524	|	114.13 	|
|	ViT_base_patch32_384	|	T4	|	fp32	|	1000	|	2	|	1	|	True	|	True	|	18.5213	|	107.98 	|
|	ViT_base_patch32_384	|	T4	|	fp32	|	1000	|	4	|	1	|	True	|	False	|	33.7056	|	118.67 	|
|	ViT_base_patch32_384	|	T4	|	fp32	|	1000	|	4	|	1	|	True	|	True	|	34.7381	|	115.15 	|
|	deeplabv3p_resnet50	|	T4	|	fp16	|	1000	|	1	|	1	|	True	|	True	|	26.1575	|	38.23 	|
|	deeplabv3p_resnet50	|	T4	|	fp16	|	1000	|	2	|	1	|	True	|	True	|	47.9256	|	41.73 	|
|	deeplabv3p_resnet50	|	T4	|	fp16	|	1000	|	4	|	1	|	True	|	True	|	95.9487	|	41.69 	|
|	deeplabv3p_resnet50	|	T4	|	fp32	|	1000	|	1	|	1	|	True	|	False	|	61.204	|	16.34 	|
|	deeplabv3p_resnet50	|	T4	|	fp32	|	1000	|	1	|	1	|	True	|	True	|	66.8809	|	14.95 	|
|	deeplabv3p_resnet50	|	T4	|	fp32	|	1000	|	2	|	1	|	True	|	False	|	119.9978	|	16.67 	|
|	deeplabv3p_resnet50	|	T4	|	fp32	|	1000	|	2	|	1	|	True	|	True	|	133.6688	|	14.96 	|
|	deeplabv3p_resnet50	|	T4	|	fp32	|	1000	|	4	|	1	|	True	|	False	|	237.8694	|	16.82 	|
|	deeplabv3p_resnet50	|	T4	|	fp32	|	1000	|	4	|	1	|	True	|	True	|	266.9613	|	14.98 	|
|	mask_rcnn_r50_vd_fpn_1x_coco	|	T4	|	fp16	|	1000	|	1	|	1	|	True	|	True	|	40.6577	|	24.60 	|
|	mask_rcnn_r50_vd_fpn_1x_coco	|	T4	|	fp32	|	1000	|	1	|	1	|	True	|	False	|	122.1686	|	8.19 	|
|	mask_rcnn_r50_vd_fpn_1x_coco	|	T4	|	fp32	|	1000	|	1	|	1	|	True	|	True	|	101.93	|	9.81 	|
|	ssdlite_mobilenet_v1_300_coco	|	T4	|	fp32	|	1000	|	1	|	1	|	True	|	False	|	7.2467	|	137.99 	|
|	ssdlite_mobilenet_v1_300_coco	|	T4	|	fp32	|	1000	|	2	|	1	|	True	|	False	|	9.0278	|	221.54 	|
|	ssdlite_mobilenet_v1_300_coco	|	T4	|	fp32	|	1000	|	4	|	1	|	True	|	False	|	12.0645	|	331.55 	|
|	yolov3_darknet53_270e_coco_upload	|	T4	|	fp16	|	1000	|	1	|	1	|	True	|	True	|	20.6326	|	48.47 	|
|	yolov3_darknet53_270e_coco_upload	|	T4	|	fp16	|	1000	|	2	|	1	|	True	|	True	|	41.5202	|	48.17 	|
|	yolov3_darknet53_270e_coco_upload	|	T4	|	fp16	|	1000	|	4	|	1	|	True	|	True	|	80.3059	|	49.81 	|
|	yolov3_darknet53_270e_coco_upload	|	T4	|	fp32	|	1000	|	1	|	1	|	True	|	False	|	52.5364	|	19.03 	|
|	yolov3_darknet53_270e_coco_upload	|	T4	|	fp32	|	1000	|	1	|	1	|	True	|	True	|	44.1216	|	22.66 	|
|	yolov3_darknet53_270e_coco_upload	|	T4	|	fp32	|	1000	|	2	|	1	|	True	|	False	|	99.3271	|	20.14 	|
|	yolov3_darknet53_270e_coco_upload	|	T4	|	fp32	|	1000	|	2	|	1	|	True	|	True	|	85.4666	|	23.40 	|
|	yolov3_darknet53_270e_coco_upload	|	T4	|	fp32	|	1000	|	4	|	1	|	True	|	False	|	196.4797	|	20.36 	|
|	yolov3_darknet53_270e_coco_upload	|	T4	|	fp32	|	1000	|	4	|	1	|	True	|	True	|	183.9448	|	21.75 	|