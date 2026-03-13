# Plan 018：Optical Flow 项目整理与结构重建

> **状态**: 🔄 进行中 | **日期**: 2026-03-13
>
> **注记 / 2026-03-13 更新**:
> 本计划创建时的主线基线是 `144x192`。
> 在同日后续 arena 实测中，当前默认部署基线已更新为 `157x203 -> 160x208`。
> 因此本计划正文中出现的 `144x192`，除非明确写的是“历史验证结果”或“导出目录名”，否则应视为过期表述，并以 `docs/DEPLOYMENT.md`、`docs/MINIMAL_DEPLOYMENT.md`、`docs/KNOWLEDGE_BASE.md` 为准。

## 执行进度

- **2026-03-13 / Phase A 已完成**
  - 新增 `plan/README.md` 作为整理入口，明确不修改根目录 `README.md`
  - 新增 `docs/START_HERE.md` 与 `docs/MINIMAL_DEPLOYMENT.md`，为 0-context agent 提供明确接管入口和最短部署路径
  - 更新 `plan-000-context-index.md`，固定 `144x192` 为当前主线
  - 更新 `docs/KNOWLEDGE_BASE.md`，移除“150x200 仍是当前主线”的混乱表述
  - 新增 `docs/DEPLOYMENT.md`，只保留当前有效部署链路
- **2026-03-13 / Phase C 已完成**
  - 已将 `144x192` 模型导出逻辑复制进 `tools/model_export/optical_flow_144x192/`
  - 新增 `scripts/export_optical_flow_144x192.sh` 作为仓库内唯一推荐导出入口
  - 新增 `docs/MODEL_EXPORT.md`，明确导出入口与外部 checkpoint/calibration 依赖边界
  - 已在板端验证仓库内导出的 `144x192` 模型可直接产生光流输出
- **2026-03-13 / Phase B 已完成**
  - `optical_cam_oflow` 主线入口文件已重命名为 `optical_flow_app.*` 与 `cvapp_optical_flow.*`
  - 主线宏已切到 `OPTICAL_FLOW_MODEL_FLASH_ADDR`
  - 主线协议别名已切到 `struct_optical_flow_algoResult` 与 `DATA_TYPE_META_OPTICAL_FLOW_DATA`
  - `build_context_snapshot.sh`、活跃部署文档与 0-context 入口已同步到新命名
- **2026-03-13 / Phase D 已完成**
  - `plan-001` 到 `plan-017` 已迁入 `plan/archive/optical-flow-debug-2026Q1/`
  - 归档计划已统一增加“historical debugging”警告
  - 根层 `plan/` 仅保留当前入口、总索引和活动计划
- **2026-03-13 / Runtime memory update**
  - `optical_cam_oflow` 已改为动态 arena 预算：先分配 `curr/prev` frame buffers 和 sensor buffers，再把剩余 SRAM 分配给 TFLM arena
  - `cvapp_optical_flow.cpp` 启动日志现在会打印 `frame_buffers`、`sensor+other`、`arena_budget`、`remaining_before_arena`、`remaining_after_arena`
  - bilinear 模型上板验证结果：
    - `172x224 -> 176x224` 可启动并进入 `INVOKE`
    - `172x228 -> 176x240` 在 `AllocateTensors()` 失败，报 `Requested: 1520720, available 1421976, missing: 98744`
  - 当前 bilinear 上机边界约为 `172x224`
- **2026-03-13 / Viz runtime budget update**
  - `viz` 已从静态 `.bss.NoInit` 改为运行期动态分配，并按模型输出尺寸预留，属于必需预算项
  - 当前内存分账顺序为：`frame_buffers -> sensor+other -> viz_buffers -> arena`
  - `cvapp_optical_flow.cpp` 启动日志现在会打印 `viz_buffers`
  - 实机验证：`172x224 -> 176x224` 在新预算下仍可启动，且 `INVOKE resolution=[224,176]`，不再依赖编译期 `FLOW_VIZ_MAX_*`
- **下一阶段**
  - 进入残余清理：场景目录名、`optical_sd*` 分支旧命名与更深层兼容遗留
  - 后续独立问题：继续收敛文档和默认主线，避免 `157x203`、`172x224 bilinear`、历史 `144x192` 在入口文档中混淆
---

## 1. 背景与本计划目标

当前项目已经完成了 **光流模型的量化、部署和可视化闭环**，但仓库结构、命名、文档和部署入口仍然混杂着大量调试阶段遗留：

- 主线事实已经确定为 **`144x192` 光流模型**，但代码和文档中同时残留了 `150x200` 实验状态与若干过期结论。
- 代码主路径仍大量沿用 `yolo` 命名，已经与实际功能不符。
- 模型导出脚本仍依赖外部仓库 `@MCUFlowNet/EdgeFlowNet`，不利于项目内聚和长期维护。
- `plan/` 中保存了大量有价值的失败经验，但其中一部分结论已经被后续验证推翻，不适合作为“当前事实”继续裸露在主入口中。
- 后续还会做别的项目，因此当前仓库需要从“调试场”整理为“可维护项目模板”。

**本计划的目标**不是立刻改完所有文件，而是给出一条可执行的整理路线，把当前仓库收敛为：

1. **单一可信主线**：默认只承认 `144x192` 是当前有效部署基线。
2. **命名与结构一致**：文件、目录、宏、脚本名称与“光流”一致。
3. **部署入口单一**：模型导出、烧录、验证都能在本仓库内完成。
4. **文档层次清楚**：当前事实、操作手册、历史调试分层保存。
5. **失败经验保留但隔离**：旧实验不丢失，但不再污染主入口。

---

## 2. 基线冻结（整理前提）

在开始重构前，必须先冻结以下“当前正确事实”，作为后续所有整理动作的判断基准：

- **部署主线模型**：`144x192`。
- **有效模型 I/O**：`in(h=144,w=192,c=6)`, `out(h=144,w=192,c=2)`。
- **有效量化参数**：`scale ≈ 0.49942484`, `zp = -1`。
- **有效显示行为**：`INVOKE resolution = [192, 144]`，输出为光流可视化图，而非相机回退图。
- **无效实验状态**：当前 `150x200 -> 160x208` 模型会触发可视化 fallback，不作为整理后的默认主线。

这一步的要求是：后续任何命名重构、目录迁移、文档清理，都不得破坏这条 `144x192` 主线。

---

## 3. 总体整理策略

本次整理按 **四层收敛** 执行：

### Phase A：先纠正“事实入口”

目标：让新读者先看到正确结论，而不是历史噪音，同时**不修改原 Seeed 根目录 `README.md`**。

动作：

- 保持根目录 `README.md` 原样不动。
- 在 `plan/` 下新增 `README.md`，作为本项目整理阶段的阅读入口和“当前主线说明”。
- 更新 [plan-000-context-index.md](file:///home/enmin/Seeed_Grove_Vision_AI_Module_V2/plan/plan-000-context-index.md)，把 `144x192` 固定为主线，并将本计划设为当前活动计划。
- 更新 [docs/KNOWLEDGE_BASE.md](file:///home/enmin/Seeed_Grove_Vision_AI_Module_V2/docs/KNOWLEDGE_BASE.md)，删除或标注已失效的 `150x200`/旧内存结论。
- 新增一份“当前有效部署说明”，只保留可复现主链路：
  - 模型导出
  - 固件构建
  - 烧录
  - 串口验证
  - 可视化验证

验收标准：

- 一个新读者只读 `plan/README + plan-000 + 当前部署说明`，就能知道项目是什么、当前哪条链路是有效的、怎样复现。

### Phase B：命名与目录去 YOLO 化

目标：让代码名称反映真实业务语义。

动作：

- 盘点所有当前仍沿用 `yolo` 语义的文件、宏、脚本、变量、编译目标。
- 建立 **“旧名 → 新名”迁移表**，优先处理主线入口：
  - `cvapp_optical_flow.cpp/.h`
  - `optical_flow_app.mk`
  - `optical_flow_app.c/.h`
  - `OPTICAL_FLOW_MODEL_FLASH_ADDR`
- 采用 **两阶段重命名**：
  - 第 1 阶段：先引入光流新名，并保留旧名兼容层或注释别名，保证可编译。
  - 第 2 阶段：验证通过后再移除旧名。
- 目录命名统一为 `optical_flow` 语义，避免继续混用 `ob` / `yolo` / `oflow` 三套口径。

建议目标命名方向：

- `cvapp_optical_flow.cpp`
- `optical_flow_app.mk`
- `OPTICAL_FLOW_MODEL_FLASH_ADDR`
- `app/scenario_app/optical_flow/`

验收标准：

- 主链路入口文件与宏名称中，不再出现误导性的 `yolo` 词汇。
- 仓库搜索 `rg -n "yolo"` 后，只剩兼容层、历史说明或明确标注为旧遗留的地方。

### Phase C：模型导出链路内聚到本仓库

目标：让 `Seeed_Grove_Vision_AI_Module_V2` 自己拥有完整的模型导出能力，不再依赖外部项目路径作为操作入口。

动作：

- 将 `@MCUFlowNet/EdgeFlowNet` 当前实际使用的 `144x192` 导出脚本和最小依赖**复制**进入本仓库。
- 新建仓库内模型导出区域，建议形态如下：

```text
tools/
  model_export/
    optical_flow_144x192/
      README.md
      run_export.py
      network/
      misc/
      vela/
```

- 明确“复制进仓库”的边界：
  - 复制实际运行所需脚本与本地 Python 逻辑
  - 不盲目整仓搬运整个 `EdgeFlowNet`
  - 对仍然外部依赖的数据集、checkpoint 路径，在文档中显式声明
- 在仓库内新增单一导出命令入口，例如：
  - `scripts/export_optical_flow_144x192.sh`
- 把导出得到的模型产物与命名规范固定下来，例如：
  - `model_zoo/optical_flow/144x192/optical_flow_144x192_vela.tflite`

验收标准：

- 新用户只进入本仓库，就能知道如何导出 `144x192` 模型。
- 仓库内存在明确的“脚本来源说明”和“与外部训练仓的边界说明”。
- 部署命令不再依赖手工引用 `MCUFlowNet/EdgeFlowNet/sramTest/run_sram_test.py`。

### Phase D：历史计划与调试文档归档

目标：保留失败经验，但不让它们继续充当“当前事实”。

动作：

- 将 `plan/` 划分为两层：
  - `plan/active/` 或保持根层只放当前主计划与总索引
  - `plan/archive/` 用于历史调试记录
- 将大量已完成、已过期、已被推翻的调试计划迁移到归档子目录，例如：

```text
plan/
  plan-000-context-index.md
  plan-018-optical-flow-project-reorganization.md
  archive/
    optical-flow-debug-2026Q1/
      plan-004-...
      plan-005-...
      ...
```

- 为归档区新增 `README.md` 或 `archive-index.md`，给每个旧计划加状态标签：
  - `historical-valid`
  - `superseded`
  - `partially-invalid`
  - `reference-only`
- 对被后续事实推翻的计划，不删除内容，但在文件开头加清晰警告：
  - “此文档保留调试过程，结论已被后续计划推翻，不应作为当前事实使用。”

验收标准：

- `plan/` 根层只保留真正需要新会话优先读取的少量文件。
- 旧计划仍可追溯，但不会再与当前主线混淆。

---

## 4. 目标目录结构（整理完成后的理想形态）

建议的项目顶层分层如下：

```text
Seeed_Grove_Vision_AI_Module_V2/
  README.md  # 保持原 Seeed 内容，不在本轮整理中改写
  docs/
    DEPLOYMENT.md
    MODEL_EXPORT.md
    ARCHITECTURE.md
    KNOWLEDGE_BASE.md
    history/
  plan/
    README.md
    plan-000-context-index.md
    plan-018-optical-flow-project-reorganization.md
    archive/
  scripts/
    export_optical_flow_144x192.sh
    deploy_optical_flow.sh
    verify_optical_flow_uart.sh
  tools/
    model_export/
      optical_flow_144x192/
  model_zoo/
    optical_flow/
      144x192/
  EPII_CM55M_APP_S/
    app/scenario_app/optical_flow/
```

这个结构的原则是：

- 根目录 `README.md` 保留上游仓库语义
- `plan/README.md` 放本项目当前接管入口
- `docs/` 放稳定说明文档
- `plan/` 放项目治理和阶段计划
- `scripts/` 放一键操作入口
- `tools/model_export/` 放导出逻辑
- `model_zoo/optical_flow/144x192/` 放主线模型资产

---

## 5. 建议执行顺序

为避免整理过程中把现有可用链路改坏，执行顺序固定如下：

1. **先修文档，不动主链路代码**
   - 更新 `plan-000`
   - 更新 `KNOWLEDGE_BASE`
   - 新建当前部署说明
2. **再收模型导出链路**
   - 先复制脚本进仓库
   - 确认新入口可运行
   - 再切换文档引用
3. **再做命名重构**
   - 优先宏、脚本、文件名
   - 最后再动目录名
4. **最后归档旧计划**
   - 等新的主线文档和命名稳定后再搬迁历史计划

原因：

- 文档先行可以立即降低混乱度。
- 导出链路先内聚，后续命名重构才不会继续被外部路径绑住。
- 历史计划归档必须放到最后，否则会在整理过程中丢失上下文引用。

---

## 6. 风险控制

### 风险 A：大规模重命名导致构建脚本断裂

控制策略：

- 每轮只重命名一层入口。
- 保留兼容别名一轮验证周期。
- 每次改名后都执行一次 `build + flash + UART keyword + frame extraction`。

### 风险 B：复制导出脚本后形成“双真相”

控制策略：

- 一旦仓库内导出脚本验证通过，外部路径只保留为“来源说明”，不再写入主文档。
- 文档中明确标注“唯一受支持的导出入口”。

### 风险 C：历史计划归档后链接失效

控制策略：

- 先生成 archive index，再迁移文件。
- 同步修复 `plan-000` 和必要的绝对路径链接。

---

## 7. 本计划的交付物

本计划执行完成后，仓库应至少具备以下交付物：

- 一份可信的 `plan/README`
- 一份只描述当前有效链路的部署文档
- 一套仓库内可调用的 `144x192` 模型导出脚本
- 一套与“光流”一致的主线命名
- 一个归档后的 `plan/` 结构
- 一份归档索引，说明哪些历史计划还能参考、哪些已经失效

---

## 8. 完成判定（Done Definition）

当以下条件全部满足时，本轮整理视为完成：

- 默认文档不再把 `150x200` 误写成当前主线。
- 主入口命名不再误导为 `yolo` 项目。
- 模型导出与部署流程在本仓库内闭环。
- 新用户可以在不阅读历史调试日志的情况下完成一次 `144x192` 部署。
- 历史失败经验仍然保留，但已经被归档并加上状态说明。

---

## 9. 下一步执行方式

本计划建议拆成三个连续执行回合：

1. **回合 1：事实与文档收敛**
2. **回合 2：模型导出链路内聚 + 部署入口统一**
3. **回合 3：命名重构 + 历史归档**

> 这样可以保证每一回合结束后，仓库都仍然处于可构建、可部署、可验证状态。
