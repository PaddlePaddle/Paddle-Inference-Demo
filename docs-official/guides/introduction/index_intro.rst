##########
Paddle Inference 简介
##########

Paddle Inference 是飞桨的原生推理库，提供服务器端的高性能推理能力。由于 Paddle Inference 能力直接基于飞桨的训练算子，因此它支持飞桨训练出的所有模型的推理。

Paddle Inference 功能特性丰富，性能优异，针对不同平台不同的应用场景进行了深度的适配优化，做到高吞吐、低时延，保证了飞桨模型在服务器端即训即用，快速部署。如果Paddle Inference可以满足您的部署需求，建议您先阅览 **推理流程** 了解标准的推理应用开发流程。

- `推理流程 <workflow.html>`_

如果您不确定 Paddle Inference 是否是最佳选择，您可以阅览 **选择推理引擎** 来选择适合您的部署方式。

- `选择推理引擎 <summary.html>`_

..  toctree::
    :hidden:

    summary.md
    workflow.md
    design.rst
    roadmap.md
