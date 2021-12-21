# Predictor 方法

Paddle Inference 的预测器，由 `createPaddlePredictor` 根据 `Config` 进行创建。用户可以根据 Predictor 提供的接口设置输入数据、执行模型预测、获取输出等。

## 创建 Predictor

相关方法定义如下：

```java
// 构造方法，创建Predictor
// 参数：config - 用于构建 Predictor 的配置信息
// 返回：Predictor对象
public static Predictor createPaddlePredictor(Config config);
```

代码示例：

```java
import com.baidu.paddle.inference.Config;

public class Hello {

    static {
        System.load(".so文件路径");
    }

    public static void main(String[] args) {
        // 创建 Config 对象
        Config config = new Config();
        
        // 设置预测模型路径
        config.setCppModel("model_file", "params_file");
        
        // 根据 Config 构建预测执行对象 Predictor
        Predictor predictor = Predictor.createPaddlePredictor(config);
        
        System.out.println(predictor.getInputNum());
    }
}

```

## 输入输出与执行预测

API 定义如下：

```java
// 获取模型输入 Tensor 的数量
// 参数：无
// 返回：long - 模型输入 Tensor 的数量
public long getInputNum()

// 获取模型输出 Tensor 的数量
// 参数：无
// 返回：long - 模型输出 Tensor 的数量
public long getOutputNum()

// 获取输入 Tensor 名称
// 参数：long - 模型输入 Tensor 的id
// 返回：String - 输入 Tensor 名称
public String getInputNameById(long id)

// 获取输出 Tensor 名称
// 参数：long - 模型输出 Tensor 的id
// 返回：String - 输出 Tensor 名称
public String getOutputNameById(long id)

// 获取输入 handle
// 参数：name - 输入handle名称
// 返回：Tensor - 输入 handle
public Tensor getInputHandle(String name)

// 获取输出 handle
// 参数：name - 输出handle名称
// 返回：Tensor - 输出 handle
public Tensor getOutputHandle(String name)

// 执行预测
// 参数：无
// 返回：None
public boolean run()

// 释放中间Tensor
// 参数：None
// 返回：None
public void clearIntermediateTensor()

// 释放内存池中的所有临时 Tensor
// 参数：None
// 返回：None
public void tryShrinkMemory()
```

代码示例：

```java
import com.baidu.paddle.inference.Predictor;
import com.baidu.paddle.inference.Config;
import com.baidu.paddle.inference.Tensor;

public class Hello {

    static {
        System.load(".so文件路径");
    }

    public static void main(String[] args) {
        // 创建 Config 对象
        Config config = new Config();
        
        // 设置预测模型路径
        config.setCppModel("model_file", "params_file");
        
        // 根据 Config 构建预测执行对象 Predictor
        Predictor predictor = Predictor.createPaddlePredictor(config);

        // 获取输入输出 Tensor 信息
        long n = predictor.getInputNum();
        String inNames = predictor.getInputNameById(0);

        // 获取输入 Tensor 指针
        Tensor inHandle = predictor.getInputHandle(inNames);
        inHandle.reshape(4, new int[]{1, 3, 224, 224});

        float[] inData = new float[1*3*224*224];
        inHandle.copyFromCpu(inData);
        predictor.run();
        String outNames = predictor.getOutputNameById(0);
        
        // 获取输出 Tensor 指针
        Tensor outHandle = predictor.getOutputHandle(outNames);
        float[] outData = new float[outHandle.getSize()];
        outHandle.copyToCpu(outData);

        System.out.println(outData[0]);
        System.out.println(outData.length);

        outHandle.destroyNativeTensor();
        inHandle.destroyNativeTensor();
        predictor.destroyNativePredictor();
    }
}

```
