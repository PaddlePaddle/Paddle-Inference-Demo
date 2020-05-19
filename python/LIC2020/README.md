## 运行LIC2020关系抽取样例

该工程是[2020语言与智能技术竞赛：关系抽取任务](https://aistudio.baidu.com/aistudio/competition/detail/31)竞赛提供的基线模型的python预测demo，基线模型的训练相关信息可参考[LIC2020-关系抽取任务基线系统](https://aistudio.baidu.com/aistudio/projectdetail/357344)。

### 一：准备环境

请您在环境中安装1.7或以上版本的Paddle，具体的安装方式请参照[飞桨官方页面](https://www.paddlepaddle.org.cn/)的指示方式。

### 二：下载模型以及测试数据


**获取预测模型**

点击[链接](https://paddle-inference-dist.bj.bcebos.com/lic_model.tgz)下载模型，如果你想获取更多的**模型训练信息**，请访问[LIC2020-关系抽取任务基线系统](https://aistudio.baidu.com/aistudio/projectdetail/357344)。

下载模型后将其存在工程的根目录，供运行时使用。

### 三：运行预测

`data` 包含了输入测试数据，词表等数据。
`ernie/reader` 包含了数据读取等功能。
`ernie/predict.py` 包含了创建predictor，读取输入，预测，获取输出的等功能。
`script` 为该工程的直接运行脚本

运行：
```
sh script/infer.sh
```

关系抽取运行的结果为： 

```
{"text": "科库雷克(RadovanKocurek),出生于1986年2月12日，捷克国籍，身高179厘米，体重72公斤，场上位置前锋，现在效力于贾洛内足球俱乐部", "spo_list": [{"predicate": "国籍", "object_type": {"@value": "国家"}, "subject_type": "人物", "object": {"@value": "捷克"}, "subject": "科库雷克"}]}
```

### 相关链接
- [Paddle Inference使用Quick Start！]()
- [Paddle Inference Python Api使用]()

