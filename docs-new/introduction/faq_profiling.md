# 精度与性能
如果您的推理程序输出结果存在精度异常，或者您的程序性能和内存消耗不符合您的预期，可以参考以下常见问题并尝试解决。


1. Predictor是否有Profile工具。  
**答：**  `config.EnableProfile()`可以打印op耗时，请参考[API文档-Profile设置](temp-link)。

2. 同一个模型的推理耗时不稳定。  
**答：** 请按以下方向排查：
1）硬件资源（CPU、GPU等）是否没有他人抢占。
2）输入是否一致，某些模型推理时间跟输入有关，比如检测模型的候选框数量。
3）使用TensorRT时，初始的优化阶段比较耗时，可以通过少量数据warm up的方式解决。

3. 如何开启CPU预测的多线程加速。  
**答：** 请使用`config.EnableMKLDNN()`和`config.SetCpuMathLibraryNumThreads()`，请参考[API文档-CPU预测](temp-link)。