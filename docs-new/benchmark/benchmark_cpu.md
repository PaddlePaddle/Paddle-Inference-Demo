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
|	MobileNetV1	|	1000	|	1	|	False	|	fp32	|	1	|	37.7486	|	26.49104867	|
|	MobileNetV1	|	1000	|	1	|	True	|	fp32	|	1	|	15.4455	|	64.7437765	|
|	MobileNetV1	|	1000	|	2	|	False	|	fp32	|	1	|	78.1411	|	25.59472544	|
|	MobileNetV1	|	1000	|	2	|	True	|	fp32	|	1	|	31.802	|	62.88912647	|
|	MobileNetV1	|	1000	|	4	|	False	|	fp32	|	1	|	150.2198	|	26.62764829	|
|	MobileNetV1	|	1000	|	4	|	True	|	fp32	|	1	|	57.1735	|	69.96248262	|
|	MobileNetV2	|	1000	|	1	|	False	|	fp32	|	1	|	43.6175	|	22.92657764	|
|	MobileNetV2	|	1000	|	1	|	True	|	fp32	|	1	|	14.8715	|	67.24271257	|
|	MobileNetV2	|	1000	|	2	|	False	|	fp32	|	1	|	85.8639	|	23.29267597	|
|	MobileNetV2	|	1000	|	2	|	True	|	fp32	|	1	|	25.7693	|	77.61173179	|
|	MobileNetV2	|	1000	|	4	|	False	|	fp32	|	1	|	175.4801	|	22.79460748	|
|	MobileNetV2	|	1000	|	4	|	True	|	fp32	|	1	|	49.5933	|	80.65605636	|
|	ResNet101	|	1000	|	1	|	False	|	fp32	|	1	|	209.7689	|	4.767150898	|
|	ResNet101	|	1000	|	1	|	True	|	fp32	|	1	|	138.5197	|	7.219189761	|
|	ResNet101	|	1000	|	2	|	False	|	fp32	|	1	|	411.6655	|	4.858313364	|
|	ResNet101	|	1000	|	2	|	True	|	fp32	|	1	|	267.575	|	7.474539849	|
|	ResNet101	|	1000	|	4	|	False	|	fp32	|	1	|	821.0667	|	4.871711397	|
|	ResNet101	|	1000	|	4	|	True	|	fp32	|	1	|	498.7897	|	8.019411788	|
|	SwinTransformer_base_patch4_window12_384	|	1000	|	1	|	False	|	fp32	|	1	|	2476.451	|	0.403803669	|
|	SwinTransformer_base_patch4_window12_384	|	1000	|	1	|	True	|	fp32	|	1	|	4309.8916	|	0.232024397	|
|	SwinTransformer_base_patch4_window12_384	|	1000	|	2	|	False	|	fp32	|	1	|	4919.3384	|	0.406558736	|
|	SwinTransformer_base_patch4_window12_384	|	1000	|	2	|	True	|	fp32	|	1	|	8538.6084	|	0.234230205	|
|	SwinTransformer_base_patch4_window12_384	|	1000	|	4	|	False	|	fp32	|	1	|	9718.9913	|	0.411565344	|
|	SwinTransformer_base_patch4_window12_384	|	1000	|	4	|	True	|	fp32	|	1	|	17098.5246	|	0.233938313	|
|	ViT_base_patch32_384	|	1000	|	1	|	False	|	fp32	|	1	|	365.7941	|	2.733778374	|
|	ViT_base_patch32_384	|	1000	|	1	|	True	|	fp32	|	1	|	326.9727	|	3.058359306	|
|	ViT_base_patch32_384	|	1000	|	2	|	False	|	fp32	|	1	|	646.3851	|	3.094130728	|
|	ViT_base_patch32_384	|	1000	|	2	|	True	|	fp32	|	1	|	1126.7091	|	1.775081075	|
|	ViT_base_patch32_384	|	1000	|	4	|	False	|	fp32	|	1	|	1218.0988	|	3.283805878	|
|	ViT_base_patch32_384	|	1000	|	4	|	True	|	fp32	|	1	|	2187.3777	|	1.828673667	|
|	bert	|	1000	|	1	|	False	|	fp32	|	1	|	106.6469	|	9.376737627	|
|	bert	|	1000	|	1	|	True	|	fp32	|	1	|	106.6411	|	9.377247609	|
|	bert	|	1000	|	2	|	False	|	fp32	|	1	|	149.6218	|	13.36703609	|
|	bert	|	1000	|	2	|	True	|	fp32	|	1	|	136.8391	|	14.6157056	|
|	bert	|	1000	|	4	|	False	|	fp32	|	1	|	276.0263	|	14.49137274	|
|	bert	|	1000	|	4	|	True	|	fp32	|	1	|	243.8251	|	16.40520193	|
|	deeplabv3p_resnet50	|	1000	|	1	|	False	|	fp32	|	1	|	3064.0091	|	0.326369788	|
|	deeplabv3p_resnet50	|	1000	|	1	|	True	|	fp32	|	1	|	2218.0117	|	0.450854249	|
|	deeplabv3p_resnet50	|	1000	|	2	|	False	|	fp32	|	1	|	6217.048	|	0.321696085	|
|	deeplabv3p_resnet50	|	1000	|	2	|	True	|	fp32	|	1	|	4378.2782	|	0.456800575	|
|	deeplabv3p_resnet50	|	1000	|	4	|	False	|	fp32	|	1	|	12464.9701	|	0.320899286	|
|	deeplabv3p_resnet50	|	1000	|	4	|	True	|	fp32	|	1	|	8859.03	|	0.451516701	|
|	mask_rcnn_r50_vd_fpn_1x_coco	|	1000	|	1	|	False	|	fp32	|	1	|	6924.0275	|	0.144424614	|
|	mask_rcnn_r50_vd_fpn_1x_coco	|	1000	|	1	|	True	|	fp32	|	1	|	3992.9994	|	0.250438305	|
|	ssdlite_mobilenet_v1_300_coco	|	1000	|	1	|	False	|	fp32	|	1	|	88.7488	|	11.26775799	|
|	ssdlite_mobilenet_v1_300_coco	|	1000	|	1	|	True	|	fp32	|	1	|	36.2734	|	27.56841101	|
|	ssdlite_mobilenet_v1_300_coco	|	1000	|	2	|	False	|	fp32	|	1	|	164.834	|	12.13341908	|
|	ssdlite_mobilenet_v1_300_coco	|	1000	|	2	|	True	|	fp32	|	1	|	66.3129	|	30.16004427	|
|	ssdlite_mobilenet_v1_300_coco	|	1000	|	4	|	False	|	fp32	|	1	|	343.1162	|	11.65785818	|
|	ssdlite_mobilenet_v1_300_coco	|	1000	|	4	|	True	|	fp32	|	1	|	132.0374	|	30.29444688	|
|	yolov3_darknet53_270e_coco_upload	|	1000	|	1	|	False	|	fp32	|	1	|	1826.1788	|	0.547591506	|
|	yolov3_darknet53_270e_coco_upload	|	1000	|	1	|	True	|	fp32	|	1	|	1160.4247	|	0.86175346	|
|	yolov3_darknet53_270e_coco_upload	|	1000	|	2	|	False	|	fp32	|	1	|	3715.4342	|	0.538295093	|
|	yolov3_darknet53_270e_coco_upload	|	1000	|	2	|	True	|	fp32	|	1	|	2318.9167	|	0.862471688	|
|	yolov3_darknet53_270e_coco_upload	|	1000	|	4	|	False	|	fp32	|	1	|	7251.0338	|	0.551645477	|
|	yolov3_darknet53_270e_coco_upload	|	1000	|	4	|	True	|	fp32	|	1	|	4635.9207	|	0.862827529	|
