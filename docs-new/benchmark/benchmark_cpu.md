# CPU 性能数据

## 测试条件

- 测试模型
	- MobileNetV1
	- MobileNetV2
	- ResNet101
	- bert
	- ViT_base_patch32_384
  	- ssdlite_mobilenet_v1_300_coco

- 测试机器信息
	- Intel(R) Xeon(R) Gold 6271C CPU @ 2.60GHz
	- 内存 250G
- 测试说明
	- 测试 PaddlePaddle 版本：v2.3
	- warmup=10，repeats=1000，统计平均时间，单位为 ms。
	- cpu_math_library_num_threads=1，num_samples=1000。

## 数据

|	model_name	|	batch_size	|	enable_mkldnn	|	precision	|	avg_latency	|	qps	|
|-|-|-|-|-|-|
|	MobileNetV1	|	1	|	False	|	fp32	|	37.7486	|	26.49 	|
|	MobileNetV1	|	1	|	True	|	fp32	|	15.4455	|	64.74 	|
|	MobileNetV1	|	2	|	False	|	fp32	|	78.1411	|	25.59 	|
|	MobileNetV1	|	2	|	True	|	fp32	|	31.802	|	62.89 	|
|	MobileNetV1	|	4	|	False	|	fp32	|	150.2198	|	26.63 	|
|	MobileNetV1	|	4	|	True	|	fp32	|	57.1735	|	69.96 	|
|	MobileNetV2	|	1	|	False	|	fp32	|	43.6175	|	22.93 	|
|	MobileNetV2	|	1	|	True	|	fp32	|	14.8715	|	67.24 	|
|	MobileNetV2	|	2	|	False	|	fp32	|	85.8639	|	23.29 	|
|	MobileNetV2	|	2	|	True	|	fp32	|	25.7693	|	77.61 	|
|	MobileNetV2	|	4	|	False	|	fp32	|	175.4801	|	22.79 	|
|	MobileNetV2	|	4	|	True	|	fp32	|	49.5933	|	80.66 	|
|	ResNet101	|	1	|	False	|	fp32	|	209.7689	|	4.77 	|
|	ResNet101	|	1	|	True	|	fp32	|	138.5197	|	7.22 	|
|	ResNet101	|	2	|	False	|	fp32	|	411.6655	|	4.86 	|
|	ResNet101	|	2	|	True	|	fp32	|	267.575	|	7.47 	|
|	ResNet101	|	4	|	False	|	fp32	|	821.0667	|	4.87 	|
|	ResNet101	|	4	|	True	|	fp32	|	498.7897	|	8.02 	|
|	ViT_base_patch32_384	|	1	|	False	|	fp32	|	365.7941	|	2.73 	|
|	ViT_base_patch32_384	|	1	|	True	|	fp32	|	326.9727	|	3.06 	|
|	ViT_base_patch32_384	|	2	|	False	|	fp32	|	646.3851	|	3.09 	|
|	ViT_base_patch32_384	|	2	|	True	|	fp32	|	1126.7091	|	1.78 	|
|	ViT_base_patch32_384	|	4	|	False	|	fp32	|	1218.0988	|	3.28 	|
|	ViT_base_patch32_384	|	4	|	True	|	fp32	|	2187.3777	|	1.83 	|
|	bert	|	1	|	False	|	fp32	|	106.6469	|	9.38 	|
|	bert	|	1	|	True	|	fp32	|	106.6411	|	9.38 	|
|	bert	|	2	|	False	|	fp32	|	149.6218	|	13.37 	|
|	bert	|	2	|	True	|	fp32	|	136.8391	|	14.62 	|
|	bert	|	4	|	False	|	fp32	|	276.0263	|	14.49 	|
|	bert	|	4	|	True	|	fp32	|	243.8251	|	16.41 	|
|	ssdlite_mobilenet_v1_300_coco	|	1	|	False	|	fp32	|	88.7488	|	11.27 	|
|	ssdlite_mobilenet_v1_300_coco	|	1	|	True	|	fp32	|	36.2734	|	27.57 	|
|	ssdlite_mobilenet_v1_300_coco	|	2	|	False	|	fp32	|	164.834	|	12.13 	|
|	ssdlite_mobilenet_v1_300_coco	|	2	|	True	|	fp32	|	66.3129	|	30.16 	|
|	ssdlite_mobilenet_v1_300_coco	|	4	|	False	|	fp32	|	343.1162	|	11.66 	|
|	ssdlite_mobilenet_v1_300_coco	|	4	|	True	|	fp32	|	132.0374	|	30.29 	|