# 多模型部署协议

## Profile

JSON schema_version=1，每个明确模型/应用组合一份。相对模型路径以仓库根为基准。profile命令验证结构与文件哈希。

- mode=observe：只描述观察预期，不允许构建/打包/刷写。
- mode=deploy：选择真实SDK app，指定模型SHA256、相机、Flash布局、I/O和成功/失败条件。
- verify.required：必须出现的应用标志；forbidden_regex：错误模式。
- input_shape/output_shape：可选，用于输出model io日志的固件。
- min_frames/frame_resolution：可选图像验收。无图像的数值应用设min_frames=0并使用明确应用结果标志；其他协议需扩展解析器。

参考 [157×203配置](../deployments/profiles/optical-flow-157x203.json)，它固定仓库历史模型，不是当前板上172×224的身份。profile不能代替C/C++的预处理、算子注册、相机驱动或任务输出适配。新模型须核对这些接口。

## 构建（不访问USB）

Linux/WSL须安装make、Arm GNU 13.2.Rel1（gcc 13.2.1）、flock及SDK打包器依赖并加入PATH。Windows build调用默认WSL。

```text
python tools/grovectl.py build --profile deployments/profiles/optical-flow-157x203.json --dry-run
python tools/grovectl.py build --profile deployments/profiles/optical-flow-157x203.json --runs /path/to/runs
```

每次clean，避免跨app复用对象；同checkout有构建锁。保存日志、ELF、IMG、编译器/Git状态。dry-run不能代替真实编译验收。

## 打包与刷写

```text
python tools/grovectl.py package --profile deployments/profiles/optical-flow-157x203.json --firmware /path/to/firmware.img --out /path/to/new-bundle
python tools/grovectl.py flash --bundle /path/to/new-bundle/bundle.json --serial YOUR_USB_SERIAL
python tools/grovectl.py flash --bundle /path/to/new-bundle/bundle.json --serial YOUR_USB_SERIAL --execute --auto-reset
```

package复制真实固件/模型并保存SHA256。flash默认只展示计划，--execute执行已授权的刷写。核对artifact哈希、容量与区间重叠，支持多模型，v1只支持零模型偏移。布局必须先依据目标固件确认。

IMG真实分区未由静态解析器验证；任意IMG与profile并不自动兼容。优先使用该profile构建产物并审阅build-context.txt。板端版本/哈希回读尚未实现；主机哈希不等于板端身份认证。

Windows Python直接操作COM，无须先转交WSL。关闭monitor/浏览器/viewer后刷写；出现等待提示后复位，复位方式见[grove-deploy skill](../skills/grove-deploy/SKILL.md)。默认总超时300秒，失败停止，不自动换模型或擦除重试。

## 运行验收

```text
python tools/grovectl.py monitor --profile deployments/profiles/optical-flow-157x203.json --serial YOUR_USB_SERIAL --seconds 120 --auto-reset
```

收到listening后再复位（方式同上）。传输完成不等于部署验收；按[测试规范](TESTING.md)检查I/O、推理、实际帧、数值和性能。本版没有通用张量输入固件，新增任务需要相应协议适配。
