# 创建 Config

`Config` 对象相关方法用于创建预测相关配置，构建 `Predictor` 对象的配置信息，如模型路径、是否开启gpu等等。

相关方法定义如下：

```java
// 构造方法，创建Config
// 参数：None
// 返回：Config对象
public Config();
// 判断该 Config 对象是否有效
// 参数：None
// 返回：boolean
public boolean isValid();
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
        
        // 判断该 Config 对象是否有效
        config.isValid();
    }
}

```

