#######
调试与优化
#######


本章节分为以下几个小节，介绍了分析精度/性能问题的标准流程，以及进行性能优化的常用方法，供开发者参考。

- `精度核验与问题追查 <precision_tracing>`_ : 验证输出结果的精度是否满足预期，追查精度问题的原因
- `性能分析方法 <performance_analysis_profiler>`_ : 使用性能分析工具定位推理性能瓶颈
- `多线程并发推理 <multi_thread>`_ : 利用多线程并发推理来提升硬件资源利用率和吞吐


..  toctree::
    :hidden:
    
    precision_tracing.md
    performance_analysis_profiler.md
    multi_thread.md
