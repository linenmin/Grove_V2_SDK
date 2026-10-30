# 模型导出

通用部署只消费模型、固件与profile，不引入训练框架。新项目从自己的训练仓交付带身份的TFLite/Vela产物。

原光流网络、导出与校准资产在[光流项目](../projects/optical_flow/README.md)，用于历史复现。scripts/export_optical_flow_144x192.sh保留兼容，PYTHON_BIN指定研究环境；该历史名字默认导出157×203，不代表板上172×224模型。导出/量化未因本次目录整理而重新验证。
