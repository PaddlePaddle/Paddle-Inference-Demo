.. Paddle Inference documentation master file, created by
   sphinx-quickstart on Thu Feb  6 14:11:30 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Paddle Inference's documentation!
===============================================

.. toctree::
  :maxdepth: 1
  :caption: 产品介绍
  :name: product-introduction

  product_introduction/summary
  product_introduction/inference_intro

.. toctree::
  :maxdepth: 1
  :caption: 快速开始
  :name: sec-quick-start

  quick_start/workflow
  quick_start/cpp_demo
  quick_start/python_demo
  quick_start/c_demo
  quick_start/go_demo

.. toctree::
  :maxdepth: 1
  :caption: 使用方法
  :name: sec-user-guides
  
  user_guides/source_compile
  user_guides/compile_ARM
  user_guides/compile_SW
  user_guides/compile_ZHAOXIN
  user_guides/compile_MIPS
  user_guides/download_lib

.. toctree::
  :maxdepth: 1
  :caption: 性能调优
  :name: sec-optimize
  
  optimize/paddle_trt
  optimize/paddle_x86_cpu_int8
  optimize/paddle_x86_cpu_bf16


.. toctree::
  :maxdepth: 1
  :caption: 工具 
  :name: sec-tools
  
  tools/visual
  tools/x2paddle

.. toctree::
  :maxdepth: 1
  :caption: 硬件部署示例 
  :name: sec-demo
  
  demo_tutorial/x86_linux_demo
  demo_tutorial/x86_windows_demo
  demo_tutorial/paddle_xpu_infer_cn
  demo_tutorial/cuda_linux_demo
  demo_tutorial/cuda_jetson_demo
  demo_tutorial/cuda_windows_demo

.. toctree::
  :maxdepth: 1
  :caption: Benchmark
  :name: sec-benchmark
  
  benchmark/benchmark

.. toctree::
  :maxdepth: 1
  :caption: API 文档
  :name: sec-api-reference

  api_reference/cxx_api_index
  api_reference/python_api_index
  api_reference/c_api_index
  api_reference/go_api_index
  api_reference/r_api_index

.. toctree::
  :maxdepth: 1
  :caption: FAQ

  introduction/faq
  introduction/training_to_deployment
.. toctree::
  :maxdepth: 1
  :caption: Debug

  debug/performance_analysis_profiler
  debug/precision_tracing
