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

- 测试机器
	- P4
	- T4
	- cpu
	- ...
- 测试说明
	- 测试 PaddlePaddle 版本：v2.3
	- warmup=10，repeats=1000，统计平均时间，单位为 ms。

## 数据


|model_name|	batch_size|	num_samples|	device|	ir_optim|	enable_tensorrt|	enable_mkldnn|	trt_precision|	cpu_math_library_num_threads|	Average_latency|	QPS|
|-|-|-|-|-|-|-|-|-|-|-|
|bert_emb_v1-paddle|	1|	1000|	P4|	true|	true|	|	fp32|	|	8.5914|	116.396|
|bert_emb_v1-paddle|	2|	1000|	P4|	true|	false|	|		|	|15.3364	|130.409|
|bert_emb_v1-paddle|	1|	1000|	P4|	true|	false|	|		|	|8.09328	|123.559|
|bert_emb_v1-paddle|	4|	1000|	P4|	true|	false|	|		|	|27.0221	|148.027|
|bert_emb_v1-paddle|	2|	1000|	P4|	true|	true|	|  fp32	|	|14.9749	|133.556|
|deeplabv3p_xception_769_fp32-paddle|	4|	1000|	P4|	true|	false|	|		|	|458.679	|8.7207|
|deeplabv3p_xception_769_fp32-paddle|	4|	1000|	P4|	true|	true|	|	fp32|	|	379.832|	10.531|
|deeplabv3p_xception_769_fp32-paddle|	1|	1000|	P4|	true|	true|	|	fp32|	|	96.0014|	10.4165|
|deeplabv3p_xception_769_fp32-paddle|	2|	1000|	P4|	true|	true|	|	fp32|	|	193.826|	10.3185|
|deeplabv3p_xception_769_fp32-paddle|	1|	1000|	P4|	true|	false|	|		|	|114.996	|8.69596|
|deeplabv3p_xception_769_fp32-paddle|	2|	1000|	P4|	true|	false|	|		|	|227.272	|8.80004|
|faster_rcnn_r50_1x-paddle|	2|	1000|	P4|	true|	true|	|	fp32||		162.795|	12.2854|
|faster_rcnn_r50_1x-paddle|	1|	1000|	P4|	true|	true|	|	fp32||		141.49	|7.06762|
|faster_rcnn_r50_1x-paddle|	4|	1000|	P4|	true|	false|	|		||	320.018	|12.4993|
|faster_rcnn_r50_1x-paddle|	2|	1000|	P4|	true|	false|	|		||	162.685	|12.2937|
|faster_rcnn_r50_1x-paddle|	1|	1000|	P4|	true|	false|	|		||	140.516	|7.11662|
|faster_rcnn_r50_1x-paddle|	4|	1000|	P4|	true|	true|	|	fp32||		318.193|	12.571|
|mobilenet_ssd-paddle|	1|	1000|	P4|	true|	false|	||	|	5.34364|	187.138|
|mobilenet_ssd-paddle|	4|	1000|	P4|	true|	false|	||	|	10.0709|	397.185|
|mobilenet_ssd-paddle|	2|	1000|	P4|	true|	false|	||	|	6.45996|	309.6|
|mobilenet_v2-paddle|	4|	1000|	P4|	true|	true|	|	fp32|	|	3.74114	|1069.19|
|mobilenet_v2-paddle|	1|	1000|	P4|	true|	true|	|	fp32|    |	1.77892|	562.14|
|mobilenet_v2-paddle|	2|	1000|	P4|	true|	true|	|	fp32|	|	2.44298	|818.673|
|mobilenet_v2-paddle|	4|	1000|	P4|	true|	false|	|	|	|	7.19198|	556.175|
|mobilenet_v2-paddle|	2|	1000|	P4|	true|	false|	|	|	|	4.53171|	441.335|
|mobilenet_v2-paddle|	1|	1000|	P4|	true|	false|	|	|	|	3.45571|	289.376|
|resnet101-paddle|	1|	1000|	P4|	true|	false|	|		|	|13.1659|	75.9538|
|resnet101-paddle|	4|	1000|	P4|	true|	false|	|		|	|21.1129|	189.457|
|resnet101-paddle|	2|	1000|	P4|	true|	true|	|	fp32|	|11.7751|	169.849|
|resnet101-paddle|	1|	1000|	P4|	true|	true|	|	fp32|	|7.79821|	128.234|
|resnet101-paddle|	4|	1000|	P4|	true|	true|	|	fp32|	|18.3|218.58|
|resnet101-paddle|	2|	1000|	P4|	true|	false|	|		|	|15.4095| 129.79|
|unet-paddle|	4|	1000|	P4|	true|	true|	|	fp32|	|155.15	|25.7814|
|unet-paddle|	1|	1000|	P4|	true|	true|	|	fp32|	|36.8867|27.11|
|unet-paddle|	2|	1000|	P4|	true|	true|	|	fp32|	|75.5283|26.4801|
|yolov3_darknet|1|	1000|	P4|	true|	false|	|	|	|84.2696|	11.8667|
|yolov3_darknet|2|	1000|	P4|	true|	false|	|	|	|139.273|	14.3603|
|yolov3_darknet|4|	1000|	P4|	true|	false|	|	|	|208.45|	19.1893|
|yolov3_darknet|1|	1000|	P4|	true|	true|	|fp32|	|43.5201|	22.9779|
|yolov3_darknet|2|	1000|	P4|	true|	true|	|fp32|	|86.456|	23.1331|
|yolov3_darknet|4|	1000|	P4|	true|	true|	|fp32|	|170.954|	23.3981|