# Tensor 方法

Tensor 是 Paddle Inference 的数据组织形式，用于对底层数据进行封装并提供接口对数据进行操作，包括设置 Shape、数据、LoD 信息等。

**注意：** 应使用 `Predictor` 的 `getInputHandle` 和 `getOutputHandle` 接口获取输入输出 `Tensor`。

Tensor 的 API 定义如下：

```java
// 获取 Tensor 维度信息
// 参数：无
// 返回：int[] - 包含 Tensor 维度信息的int数组
public int[] getShape()

// 设置 Tensor 维度信息
// 参数：dim_num - 维度信息的长度
//      shape - 维度信息数组
// 返回：None
public void reshape(int dim_num, int[] shape)

// 获取 Tensor 名称
// 参数：无
// 返回：string - Tensor 名称
public String getName()

// 设置 Tensor 数据
// 参数：obj - Tensor 数据
// 返回：None
public void copyFromCpu(Object obj)

// 获取 Tensor 数据
// 参数：obj - 用来存储 Tensor 数据
// 返回：None
public void copyToCpu(Object obj)
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
