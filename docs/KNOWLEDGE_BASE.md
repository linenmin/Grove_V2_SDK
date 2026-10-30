# 硬件与诊断

平台为Himax WE2，当前光流使用Ethos-U55。arena受相机、帧缓冲、可视化共同影响，不设全库固定预算。光流配置在 EPII_CM55M_APP_S/app/scenario_app/optical_cam_oflow/config/common_config.h。

当前光流0x3AB7B000映射地址对应0xB7B000烧录偏移，更换app须重新核对布局。串口921600/8N1来自本次实测与原流程；COM5不是永久身份，多个设备按序列号选择。

## 摄像头

2026-09-24用户确认相机接错，修正后OV5647、NPU和51张224×176帧恢复。SDK drivers/inc/hx_drv_iic.h中-60表示I²C地址无应答，不是模型算子错误。先断电检查接线、模组、供电和app相机选择，不靠重刷模型修复。

抓启动日志要先监听、后RESET；空捕获不表示恢复。串口打开可能影响复位线。

观察到旧板端172×224×6→176×224×2、scale=0.017498、zp=16，arena预算1419872B，仅适用于该次固件。精确模型身份待核验，见[观察profile](../deployments/profiles/optical-flow-observed-172x224.json)。

## 预览

INVOKE与RAW 0xFC模式传的都是JPEG，不是原始u/v。近黑图需运动/增益对照，不能直接作精度判断。[光流viewer](../projects/optical_flow/README.md)独占串口并发送模式切换，不可与monitor并行。共享串口后台服务本轮未实现。

[历史固定arena与尺寸结论](history/README.md)须连同其模型/固件条件理解。
