# 启用内存优化

相关方法定义如下：

```java
// 开启内存/显存复用，具体降低内存效果取决于模型结构
// 参数：无
// 返回：无
public void enableMemoryOptim();

// 判断是否开启内存/显存复用
// 参数：无
// 返回：boolean - 是否开启内/显存复用
public boolean memoryOptimEnabled();
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
        
        // 开启 CPU 内存优化
        config.enableMemoryOptim();
        // 通过 API 获取 CPU 是否已经开启显存优化 - true
        System.out.println("CPU Mem Optim is: ", config.memoryOptimEnabled());

        // 启用 GPU 进行预测
        config.enableUseGpu(100, 0);
        // 开启 GPU 显存优化
        config.enableMemoryOptim();
        // 通过 API 获取 GPU 是否已经开启显存优化 - true
        System.out.println("CPU Mem Optim is: ", config.memoryOptimEnabled());
    }
}
```



# Profile 设置

相关方法定义如下：

```java
// 打开 Profile，运行结束后会打印所有 OP 的耗时占比。
// 参数：无
// 返回：无
public void enableProfile();

// 判断是否开启 Profile
// 参数：无
// 返回：boolean - 是否开启 Profile
public boolean profileEnabled();
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
        
        // 打开 Profile
        config.enableProfile();
        // 判断是否开启 Profile - true
        System.out.println("Profile is: ", config.profileEnabled());
    }
}
```



# Log 设置

相关方法定义如下：

```java
// 去除 Paddle Inference 运行中的 LOG
// 参数：无
// 返回：无
public void disableGlogInfo();
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
        
        // 去除 Paddle Inference 运行中的 LOG
        config.disableGlogInfo();
    }
}
```



# 查看config配置

相关方法定义如下：

```java
// 返回config的配置信息
// 参数：无
// 返回：string - config配置信息
public String summary();
```