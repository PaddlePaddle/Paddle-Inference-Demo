# 设置模型优化方法

相关方法定义如下：

```java
// 启用 IR 优化
// 参数 flag:boolean, 是否开启 IR 优化，默认打开
// 返回：无
public void switchIrOptim(boolean flag);

// 判断是否开启 IR 优化 
// 参数：无
// 返回：boolean - 是否开启 IR 优化
public boolean irOptim();

// 设置是否在图分析阶段打印 IR，启用后会在每一个 PASS 后生成 dot 文件
// 参数：flag:boolean, 是否打印 IR，默认关闭
// 返回：无
public void switchIrDebug(boolean flag);
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

        // 开启 IR 优化
        config.switchIrOptim(true);
        
        // 开启 IR 打印
        config.switchIrDebug(true);

        // 通过 API 获取 IR 优化是否开启 - true
        System.out.println("IR Optim is: " +config.irOptim());
    }
}
```

