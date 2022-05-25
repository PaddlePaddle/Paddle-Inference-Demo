# CPU 性能数据

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
	- Intel(R) Xeon(R) Gold 6271C CPU @ 2.60GHz
	- 内存 250G
- 测试说明
	- 测试 PaddlePaddle 版本：v2.3
	- warmup=10，repeats=1000，统计平均时间，单位为 ms。

## 数据

|	model_name	|	num_samples	|	batch_size	|	enable_mkldnn	|	precision	|	cpu_math_library_num_threads	|	avg_latency	|	qps	|
|-|-|-|-|-|-|-|-|
|	MobileNetV1	|	1000	|	1	|	False	|	fp32	|	1	|	37.7486	|	26.49 	|
|	MobileNetV1	|	1000	|	1	|	True	|	fp32	|	1	|	15.4455	|	64.74 	|
|	MobileNetV1	|	1000	|	2	|	False	|	fp32	|	1	|	78.1411	|	25.59 	|
|	MobileNetV1	|	1000	|	2	|	True	|	fp32	|	1	|	31.802	|	62.89 	|
|	MobileNetV1	|	1000	|	4	|	False	|	fp32	|	1	|	150.2198	|	26.63 	|
|	MobileNetV1	|	1000	|	4	|	True	|	fp32	|	1	|	57.1735	|	69.96 	|
|	MobileNetV2	|	1000	|	1	|	False	|	fp32	|	1	|	43.6175	|	22.93 	|
|	MobileNetV2	|	1000	|	1	|	True	|	fp32	|	1	|	14.8715	|	67.24 	|
|	MobileNetV2	|	1000	|	2	|	False	|	fp32	|	1	|	85.8639	|	23.29 	|
|	MobileNetV2	|	1000	|	2	|	True	|	fp32	|	1	|	25.7693	|	77.61 	|
|	MobileNetV2	|	1000	|	4	|	False	|	fp32	|	1	|	175.4801	|	22.79 	|
|	MobileNetV2	|	1000	|	4	|	True	|	fp32	|	1	|	49.5933	|	80.66 	|
|	ResNet101	|	1000	|	1	|	False	|	fp32	|	1	|	209.7689	|	4.77 	|
|	ResNet101	|	1000	|	1	|	True	|	fp32	|	1	|	138.5197	|	7.22 	|
|	ResNet101	|	1000	|	2	|	False	|	fp32	|	1	|	411.6655	|	4.86 	|
|	ResNet101	|	1000	|	2	|	True	|	fp32	|	1	|	267.575	|	7.47 	|
|	ResNet101	|	1000	|	4	|	False	|	fp32	|	1	|	821.0667	|	4.87 	|
|	ResNet101	|	1000	|	4	|	True	|	fp32	|	1	|	498.7897	|	8.02 	|
|	SwinTransformer_base_patch4_window12_384	|	1000	|	1	|	False	|	fp32	|	1	|	2476.451	|	0.40 	|
|	SwinTransformer_base_patch4_window12_384	|	1000	|	1	|	True	|	fp32	|	1	|	4309.8916	|	0.23 	|
|	SwinTransformer_base_patch4_window12_384	|	1000	|	2	|	False	|	fp32	|	1	|	4919.3384	|	0.41 	|
|	SwinTransformer_base_patch4_window12_384	|	1000	|	2	|	True	|	fp32	|	1	|	8538.6084	|	0.23 	|
|	SwinTransformer_base_patch4_window12_384	|	1000	|	4	|	False	|	fp32	|	1	|	9718.9913	|	0.41 	|
|	SwinTransformer_base_patch4_window12_384	|	1000	|	4	|	True	|	fp32	|	1	|	17098.5246	|	0.23 	|
|	ViT_base_patch32_384	|	1000	|	1	|	False	|	fp32	|	1	|	365.7941	|	2.73 	|
|	ViT_base_patch32_384	|	1000	|	1	|	True	|	fp32	|	1	|	326.9727	|	3.06 	|
|	ViT_base_patch32_384	|	1000	|	2	|	False	|	fp32	|	1	|	646.3851	|	3.09 	|
|	ViT_base_patch32_384	|	1000	|	2	|	True	|	fp32	|	1	|	1126.7091	|	1.78 	|
|	ViT_base_patch32_384	|	1000	|	4	|	False	|	fp32	|	1	|	1218.0988	|	3.28 	|
|	ViT_base_patch32_384	|	1000	|	4	|	True	|	fp32	|	1	|	2187.3777	|	1.83 	|
|	bert	|	1000	|	1	|	False	|	fp32	|	1	|	106.6469	|	9.38 	|
|	bert	|	1000	|	1	|	True	|	fp32	|	1	|	106.6411	|	9.38 	|
|	bert	|	1000	|	2	|	False	|	fp32	|	1	|	149.6218	|	13.37 	|
|	bert	|	1000	|	2	|	True	|	fp32	|	1	|	136.8391	|	14.62 	|
|	bert	|	1000	|	4	|	False	|	fp32	|	1	|	276.0263	|	14.49 	|
|	bert	|	1000	|	4	|	True	|	fp32	|	1	|	243.8251	|	16.41 	|
|	deeplabv3p_resnet50	|	1000	|	1	|	False	|	fp32	|	1	|	3064.0091	|	0.33 	|
|	deeplabv3p_resnet50	|	1000	|	1	|	True	|	fp32	|	1	|	2218.0117	|	0.45 	|
|	deeplabv3p_resnet50	|	1000	|	2	|	False	|	fp32	|	1	|	6217.048	|	0.32 	|
|	deeplabv3p_resnet50	|	1000	|	2	|	True	|	fp32	|	1	|	4378.2782	|	0.46 	|
|	deeplabv3p_resnet50	|	1000	|	4	|	False	|	fp32	|	1	|	12464.9701	|	0.32 	|
|	deeplabv3p_resnet50	|	1000	|	4	|	True	|	fp32	|	1	|	8859.03	|	0.45 	|
|	mask_rcnn_r50_vd_fpn_1x_coco	|	1000	|	1	|	False	|	fp32	|	1	|	6924.0275	|	0.14 	|
|	mask_rcnn_r50_vd_fpn_1x_coco	|	1000	|	1	|	True	|	fp32	|	1	|	3992.9994	|	0.25 	|
|	ssdlite_mobilenet_v1_300_coco	|	1000	|	1	|	False	|	fp32	|	1	|	88.7488	|	11.27 	|
|	ssdlite_mobilenet_v1_300_coco	|	1000	|	1	|	True	|	fp32	|	1	|	36.2734	|	27.57 	|
|	ssdlite_mobilenet_v1_300_coco	|	1000	|	2	|	False	|	fp32	|	1	|	164.834	|	12.13 	|
|	ssdlite_mobilenet_v1_300_coco	|	1000	|	2	|	True	|	fp32	|	1	|	66.3129	|	30.16 	|
|	ssdlite_mobilenet_v1_300_coco	|	1000	|	4	|	False	|	fp32	|	1	|	343.1162	|	11.66 	|
|	ssdlite_mobilenet_v1_300_coco	|	1000	|	4	|	True	|	fp32	|	1	|	132.0374	|	30.29 	|
|	yolov3_darknet53_270e_coco_upload	|	1000	|	1	|	False	|	fp32	|	1	|	1826.1788	|	0.55 	|
|	yolov3_darknet53_270e_coco_upload	|	1000	|	1	|	True	|	fp32	|	1	|	1160.4247	|	0.86 	|
|	yolov3_darknet53_270e_coco_upload	|	1000	|	2	|	False	|	fp32	|	1	|	3715.4342	|	0.54 	|
|	yolov3_darknet53_270e_coco_upload	|	1000	|	2	|	True	|	fp32	|	1	|	2318.9167	|	0.86 	|
|	yolov3_darknet53_270e_coco_upload	|	1000	|	4	|	False	|	fp32	|	1	|	7251.0338	|	0.55 	|
|	yolov3_darknet53_270e_coco_upload	|	1000	|	4	|	True	|	fp32	|	1	|	4635.9207	|	0.86 	|