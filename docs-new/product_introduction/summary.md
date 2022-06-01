# 如何选择正确的推理引擎

作为飞桨生态重要的一部分，飞桨提供了多个推理产品，完整承接深度学习模型应用的最后一公里。

整体上分，推理产品主要包括如下子产品

| 名称               | 英文表示         | 适用场景                     | 典型硬件 |
|--------------------|------------------|------------------------------|------------------------------|
| 飞桨原生推理库     | Paddle Inference | 高性能服务器端、云端推理     | X86 CPU、Nvidia GPU（包括Jetson，暂不支持Orin）、飞腾/鲲鹏、申威、兆芯、龙芯等| 
| 飞桨服务化推理框架 | [Paddle Serving]()  | 服务化部署、多模型管理等高阶功能；<br>其中的AI推理部分集成Paddle Inference | X86 CPU、Nvidia GPU、Jetson、ARM CPU、百度昆仑、华为晟腾、海光DCU | 
| 飞桨轻量化推理引擎 | [Paddle Lite ](https://paddle-lite.readthedocs.io/zh/latest/index.html)   | 移动端、物联网等             |ARM CPU/GPU/NPU、华为麒麟NPU、华为晟腾NPU、瑞芯微NPU、昆仑芯XPU、晶晨NPU、颖脉NNA 、比特大陆TPU、芯原 TIM-VX、联发科APU、百度 FPGA、Intel FPGA等|
| 飞桨前端推理引擎   | [Paddle.js](https://github.com/PaddlePaddle/Paddle.js)    | 浏览器、Node.js、小程序等中做AI推理       |浏览器：主流浏览器 <br> 小程序：百度小程序、微信小程序|


各产品在推理生态中的关系如下

![](../images/inference_ecosystem.png)
