#  使用 GPU 进行预测

**注意：**

1. Config 默认使用 CPU 进行预测，需要通过 `EnableUseGpu` 来启用 GPU 预测



## GPU 设置

相关方法定义如下：

```java
// 启用 GPU 进行预测
// 参数：long: memorySize, 初始化分配的gpu显存，以MB为单位
//      int: deviceId, 设备id
// 返回：无
public void enableUseGpu(long memorySize, int deviceId);

// 禁用 GPU 进行预测
// 参数：无
// 返回：无
public void disableGpu();

// 判断是否启用 GPU 
// 参数：无
// 返回：bool - 是否启用 GPU 
public boolean useGpu();

// 获取 GPU 的device id
// 参数：无
// 返回：int -  GPU 的device id
public int getGpuDeviceId();

// 获取 GPU 的初始显存大小
// 参数：无
// 返回：int -  GPU 的初始的显存大小
public int getMemoryPoolInitSizeMb();

// 初始化显存占总显存的百分比
// 参数：无
// 返回：float32 - 初始的显存占总显存的百分比
public float getFractionOfGpuMemoryForPool();
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
        
        // 启用 GPU 进行预测 - 初始化 GPU 显存 100M, DeivceID 为 0
        config.enableUseGpu(100, 0);
        
        // 通过 API 获取 GPU 信息
        System.out.println("Use GPU is: " + config.useGpu());
        System.out.println("GPU deivce id is: " + config.getGpuDeviceId());
        System.out.println("GPU memory size is: " + config.getMemoryPoolInitSizeMb());
        System.out.println("GPU memory frac is: " + config.getFractionOfGpuMemoryForPool());

  
    	// 禁用 GPU 进行预测
    	config.disableGpu();
  
    	// 通过 API 获取 GPU 信息 - False
        System.out.println("Use GPU is: " + config.useGpu());
    }
}
```
