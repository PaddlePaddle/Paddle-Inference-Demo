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
|	MobileNetV1	|	T4	|	fp16	|	1000	|	1	|	1	|	True	|	True	|	0.4925	|	2030.456853	|
|	MobileNetV1	|	T4	|	fp16	|	1000	|	2	|	1	|	True	|	True	|	0.7485	|	2672.010688	|
|	MobileNetV1	|	T4	|	fp16	|	1000	|	4	|	1	|	True	|	True	|	1.2914	|	3097.41366	|
|	MobileNetV1	|	T4	|	fp32	|	1000	|	1	|	1	|	True	|	False	|	2.223	|	449.8425551	|
|	MobileNetV1	|	T4	|	fp32	|	1000	|	1	|	1	|	True	|	True	|	0.8737	|	1144.557628	|
|	MobileNetV1	|	T4	|	fp32	|	1000	|	2	|	1	|	True	|	False	|	3.5571	|	562.2557701	|
|	MobileNetV1	|	T4	|	fp32	|	1000	|	2	|	1	|	True	|	True	|	1.4106	|	1417.836382	|
|	MobileNetV1	|	T4	|	fp32	|	1000	|	4	|	1	|	True	|	False	|	4.0367	|	990.9084153	|
|	MobileNetV1	|	T4	|	fp32	|	1000	|	4	|	1	|	True	|	True	|	2.5238	|	1584.911641	|
|	MobileNetV2	|	T4	|	fp16	|	1000	|	1	|	1	|	True	|	True	|	0.5926	|	1687.478907	|
|	MobileNetV2	|	T4	|	fp16	|	1000	|	2	|	1	|	True	|	True	|	0.9131	|	2190.340598	|
|	MobileNetV2	|	T4	|	fp16	|	1000	|	4	|	1	|	True	|	True	|	1.4491	|	2760.334	|
|	MobileNetV2	|	T4	|	fp32	|	1000	|	1	|	1	|	True	|	False	|	3.7645	|	265.6395272	|
|	MobileNetV2	|	T4	|	fp32	|	1000	|	1	|	1	|	True	|	True	|	1.1125	|	898.8764045	|
|	MobileNetV2	|	T4	|	fp32	|	1000	|	2	|	1	|	True	|	False	|	4.5273	|	441.764407	|
|	MobileNetV2	|	T4	|	fp32	|	1000	|	2	|	1	|	True	|	True	|	1.6682	|	1198.897015	|
|	MobileNetV2	|	T4	|	fp32	|	1000	|	4	|	1	|	True	|	False	|	6.2238	|	642.694174	|
|	MobileNetV2	|	T4	|	fp32	|	1000	|	4	|	1	|	True	|	True	|	2.819	|	1418.942888	|
|	ResNet101	|	T4	|	fp16	|	1000	|	1	|	1	|	True	|	True	|	2.1345	|	468.4937925	|
|	ResNet101	|	T4	|	fp16	|	1000	|	2	|	1	|	True	|	True	|	2.9835	|	670.3536115	|
|	ResNet101	|	T4	|	fp16	|	1000	|	4	|	1	|	True	|	True	|	4.9308	|	811.227387	|
|	ResNet101	|	T4	|	fp32	|	1000	|	1	|	1	|	True	|	False	|	15.0993	|	66.22823575	|
|	ResNet101	|	T4	|	fp32	|	1000	|	1	|	1	|	True	|	True	|	6.3175	|	158.290463	|
|	ResNet101	|	T4	|	fp32	|	1000	|	2	|	1	|	True	|	False	|	17.5788	|	113.7734089	|
|	ResNet101	|	T4	|	fp32	|	1000	|	2	|	1	|	True	|	True	|	9.251	|	216.192844	|
|	ResNet101	|	T4	|	fp32	|	1000	|	4	|	1	|	True	|	False	|	20.6774	|	193.447919	|
|	ResNet101	|	T4	|	fp32	|	1000	|	4	|	1	|	True	|	True	|	16.7459	|	238.8644385	|
|	SwinTransformer_base_patch4_window12_384	|	T4	|	fp16	|	1000	|	1	|	1	|	True	|	True	|	23.0886	|	43.31141776	|
|	SwinTransformer_base_patch4_window12_384	|	T4	|	fp16	|	1000	|	2	|	1	|	True	|	True	|	42.2748	|	47.30950826	|
|	SwinTransformer_base_patch4_window12_384	|	T4	|	fp16	|	1000	|	4	|	1	|	True	|	True	|	87.3252	|	45.8057926	|
|	SwinTransformer_base_patch4_window12_384	|	T4	|	fp32	|	1000	|	1	|	1	|	True	|	False	|	42.5669	|	23.49243191	|
|	SwinTransformer_base_patch4_window12_384	|	T4	|	fp32	|	1000	|	1	|	1	|	True	|	True	|	43.5075	|	22.98454289	|
|	SwinTransformer_base_patch4_window12_384	|	T4	|	fp32	|	1000	|	2	|	1	|	True	|	False	|	84.5615	|	23.65142529	|
|	SwinTransformer_base_patch4_window12_384	|	T4	|	fp32	|	1000	|	2	|	1	|	True	|	True	|	87.5455	|	22.84526332	|
|	SwinTransformer_base_patch4_window12_384	|	T4	|	fp32	|	1000	|	4	|	1	|	True	|	False	|	169.1416	|	23.64882442	|
|	SwinTransformer_base_patch4_window12_384	|	T4	|	fp32	|	1000	|	4	|	1	|	True	|	True	|	173.796	|	23.01548942	|
|	ViT_base_patch32_384	|	T4	|	fp16	|	1000	|	1	|	1	|	True	|	True	|	4.923	|	203.1281739	|
|	ViT_base_patch32_384	|	T4	|	fp16	|	1000	|	2	|	1	|	True	|	True	|	7.5347	|	265.4385709	|
|	ViT_base_patch32_384	|	T4	|	fp16	|	1000	|	4	|	1	|	True	|	True	|	12.899	|	310.1015583	|
|	ViT_base_patch32_384	|	T4	|	fp32	|	1000	|	1	|	1	|	True	|	False	|	10.2415	|	97.64194698	|
|	ViT_base_patch32_384	|	T4	|	fp32	|	1000	|	1	|	1	|	True	|	True	|	10.8246	|	92.38216655	|
|	ViT_base_patch32_384	|	T4	|	fp32	|	1000	|	2	|	1	|	True	|	False	|	17.524	|	114.1291942	|
|	ViT_base_patch32_384	|	T4	|	fp32	|	1000	|	2	|	1	|	True	|	True	|	18.5213	|	107.9837808	|
|	ViT_base_patch32_384	|	T4	|	fp32	|	1000	|	4	|	1	|	True	|	False	|	33.7056	|	118.6746416	|
|	ViT_base_patch32_384	|	T4	|	fp32	|	1000	|	4	|	1	|	True	|	True	|	34.7381	|	115.1473454	|
|	deeplabv3p_resnet50	|	T4	|	fp16	|	1000	|	1	|	1	|	True	|	True	|	26.1575	|	38.22995317	|
|	deeplabv3p_resnet50	|	T4	|	fp16	|	1000	|	2	|	1	|	True	|	True	|	47.9256	|	41.73135026	|
|	deeplabv3p_resnet50	|	T4	|	fp16	|	1000	|	4	|	1	|	True	|	True	|	95.9487	|	41.6889442	|
|	deeplabv3p_resnet50	|	T4	|	fp32	|	1000	|	1	|	1	|	True	|	False	|	61.204	|	16.33880139	|
|	deeplabv3p_resnet50	|	T4	|	fp32	|	1000	|	1	|	1	|	True	|	True	|	66.8809	|	14.9519519	|
|	deeplabv3p_resnet50	|	T4	|	fp32	|	1000	|	2	|	1	|	True	|	False	|	119.9978	|	16.66697223	|
|	deeplabv3p_resnet50	|	T4	|	fp32	|	1000	|	2	|	1	|	True	|	True	|	133.6688	|	14.96235472	|
|	deeplabv3p_resnet50	|	T4	|	fp32	|	1000	|	4	|	1	|	True	|	False	|	237.8694	|	16.81595027	|
|	deeplabv3p_resnet50	|	T4	|	fp32	|	1000	|	4	|	1	|	True	|	True	|	266.9613	|	14.98344517	|
|	mask_rcnn_r50_vd_fpn_1x_coco	|	T4	|	fp16	|	1000	|	1	|	1	|	True	|	True	|	40.6577	|	24.59558706	|
|	mask_rcnn_r50_vd_fpn_1x_coco	|	T4	|	fp32	|	1000	|	1	|	1	|	True	|	False	|	122.1686	|	8.185409344	|
|	mask_rcnn_r50_vd_fpn_1x_coco	|	T4	|	fp32	|	1000	|	1	|	1	|	True	|	True	|	101.93	|	9.810654371	|
|	ssdlite_mobilenet_v1_300_coco	|	T4	|	fp32	|	1000	|	1	|	1	|	True	|	False	|	7.2467	|	137.9938455	|
|	ssdlite_mobilenet_v1_300_coco	|	T4	|	fp32	|	1000	|	2	|	1	|	True	|	False	|	9.0278	|	221.5379162	|
|	ssdlite_mobilenet_v1_300_coco	|	T4	|	fp32	|	1000	|	4	|	1	|	True	|	False	|	12.0645	|	331.5512454	|
|	yolov3_darknet53_270e_coco_upload	|	T4	|	fp16	|	1000	|	1	|	1	|	True	|	True	|	20.6326	|	48.46698913	|
|	yolov3_darknet53_270e_coco_upload	|	T4	|	fp16	|	1000	|	2	|	1	|	True	|	True	|	41.5202	|	48.16932481	|
|	yolov3_darknet53_270e_coco_upload	|	T4	|	fp16	|	1000	|	4	|	1	|	True	|	True	|	80.3059	|	49.80954077	|
|	yolov3_darknet53_270e_coco_upload	|	T4	|	fp32	|	1000	|	1	|	1	|	True	|	False	|	52.5364	|	19.03442185	|
|	yolov3_darknet53_270e_coco_upload	|	T4	|	fp32	|	1000	|	1	|	1	|	True	|	True	|	44.1216	|	22.66463592	|
|	yolov3_darknet53_270e_coco_upload	|	T4	|	fp32	|	1000	|	2	|	1	|	True	|	False	|	99.3271	|	20.13549172	|
|	yolov3_darknet53_270e_coco_upload	|	T4	|	fp32	|	1000	|	2	|	1	|	True	|	True	|	85.4666	|	23.40095429	|
|	yolov3_darknet53_270e_coco_upload	|	T4	|	fp32	|	1000	|	4	|	1	|	True	|	False	|	196.4797	|	20.35833727	|
|	yolov3_darknet53_270e_coco_upload	|	T4	|	fp32	|	1000	|	4	|	1	|	True	|	True	|	183.9448	|	21.74565413	|