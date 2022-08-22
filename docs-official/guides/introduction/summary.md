# 如何选择正确的推理引擎

作为飞桨生态重要的一部分，飞桨提供了多个推理产品，完整承接深度学习模型应用的最后一公里。

飞桨推理产品主要包括如下子产品：

| 名称               | 英文表示         | 适用场景                     | 典型硬件 |
|--------------------|------------------|------------------------------|------------------------------|
| 飞桨原生推理库     | Paddle Inference | 高性能服务器端、云端推理     | X86 CPU、Nvidia GPU（含Jetson系列）、飞腾/鲲鹏、申威、兆芯、龙芯、AMD GPU，海光DCU，昆仑XPU，昇腾910NPU，Graphcore IPU 等| 
| 飞桨服务化推理框架 | [Paddle Serving]()  | 服务化部署、多模型管理等高阶功能；<br>其中的AI推理部分集成Paddle Inference | x86(Intel) CPU、ARM CPU、Nvidia GPU（含Jetson系列）、昆仑 XPU、华为昇腾310/910、海光 DCU 等 | 
| 飞桨轻量化推理引擎 | [Paddle Lite ](https://paddle-lite.readthedocs.io/zh/latest/index.html)   | 移动端、物联网等             |Arm CPU、Arm Mali 系列 GPU、高通 Adreno 系列 GPU、华为麒麟 NPU、华为昇腾NPU、寒武纪MLU、瑞芯微NPU、昆仑芯XPU、晶晨NPU、Imagination NNA、比特大陆TPU、联发科APU、亿智NPU、百度 FPGA、Intel FPGA等硬件；<br> Intel OpenVINO、芯原 TIM-VX、Android NNAPI 等后端|
| 飞桨前端推理引擎   | [Paddle.js](https://github.com/PaddlePaddle/Paddle.js)    | 浏览器、Node.js、小程序等中做AI推理       |浏览器：主流浏览器 <br> 小程序：百度小程序、微信小程序|


各产品在推理生态中的关系如下：

![](../images/inference_ecosystem.png)
