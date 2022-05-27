# CPU 性能数据

## 测试条件

- 测试模型
	- MobileNetV1
	- MobileNetV2
	- ResNet50
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

|	model_name	|	batch_size	|	enable_mkldnn	|	precision	|	avg_latency	|
|-|-|-|-|-|
|	MobileNetV1	|	1	|	False	|	fp32	|	37.7486	|
|	MobileNetV1	|	1	|	True	|	fp32	|	15.4455	|
|	MobileNetV1	|	2	|	False	|	fp32	|	78.1411	|
|	MobileNetV1	|	2	|	True	|	fp32	|	31.802	|
|	MobileNetV1	|	4	|	False	|	fp32	|	150.2198	|
|	MobileNetV1	|	4	|	True	|	fp32	|	57.1735	|
|	MobileNetV2	|	1	|	False	|	fp32	|	43.6175	|
|	MobileNetV2	|	1	|	True	|	fp32	|	14.8715	|
|	MobileNetV2	|	2	|	False	|	fp32	|	85.8639	|
|	MobileNetV2	|	2	|	True	|	fp32	|	25.7693	|
|	MobileNetV2	|	4	|	False	|	fp32	|	175.4801	|
|	MobileNetV2	|	4	|	True	|	fp32	|	49.5933	|
|	ResNet50	|	1	|	False	|	fp32	|	123.8814	|
|	ResNet50	|	1	|	True	|	fp32	|	73.7028	|
|	ResNet50	|	2	|	False	|	fp32	|	232.9871	|
|	ResNet50	|	2	|	True	|	fp32	|	159.2116	|
|	ResNet50	|	4	|	False	|	fp32	|	494.5854	|
|	ResNet50	|	4	|	True	|	fp32	|	289.705	|
|	ViT_base_patch32_384	|	1	|	False	|	fp32	|	365.7941	|
|	ViT_base_patch32_384	|	1	|	True	|	fp32	|	326.9727	|
|	ViT_base_patch32_384	|	2	|	False	|	fp32	|	646.3851	|
|	ViT_base_patch32_384	|	2	|	True	|	fp32	|	1126.7091	|
|	ViT_base_patch32_384	|	4	|	False	|	fp32	|	1218.0988	|
|	ViT_base_patch32_384	|	4	|	True	|	fp32	|	2187.3777	|
|	bert	|	1	|	False	|	fp32	|	106.6469	|
|	bert	|	1	|	True	|	fp32	|	106.6411	|
|	bert	|	2	|	False	|	fp32	|	149.6218	|
|	bert	|	2	|	True	|	fp32	|	136.8391	|
|	bert	|	4	|	False	|	fp32	|	276.0263	|
|	bert	|	4	|	True	|	fp32	|	243.8251	|
|	ssdlite_mobilenet_v1_300_coco	|	1	|	False	|	fp32	|	88.7488	|
|	ssdlite_mobilenet_v1_300_coco	|	1	|	True	|	fp32	|	36.2734	|
|	ssdlite_mobilenet_v1_300_coco	|	2	|	False	|	fp32	|	164.834	|
|	ssdlite_mobilenet_v1_300_coco	|	2	|	True	|	fp32	|	66.3129	|
|	ssdlite_mobilenet_v1_300_coco	|	4	|	False	|	fp32	|	343.1162	|
|	ssdlite_mobilenet_v1_300_coco	|	4	|	True	|	fp32	|	132.0374	|