> Archived note: this file preserves historical debugging work. Do not use it as the current baseline; read `docs/DEPLOYMENT.md`, `docs/MINIMAL_DEPLOYMENT.md`, and `plan-018-optical-flow-project-reorganization.md` first.

# Plan 017: UART 二进制通道性能与瓶颈排查

> **状态**: ✅ 已关闭 (无需优化) | **日期**: 2026-02-26 → 2026-02-27

---

## 1. 调试背景

在实施 Phase 1 RAW 二进制 UART 传输后，发现主机端接收帧率（FPS）并未如预期达到 5.5 - 6.0，而是处于 3.9 - 4.6 FPS 之间，低于 Himax HTML Toolkit 显示的 5.8 FPS。

## 2. 已排除的假设

### ❌ 假设 A: 150µs/16B 限流导致延迟
- **实验**: 完全移除 `uart_send_bytes` 中的 `hx_drv_timer_cm55s_delay_us(150)` 限流
- **结果**: FPS 没有任何改善
- **结论**: 限流不是瓶颈

### ❌ 假设 B: xprintf 文本污染二进制通道
- **实验**: 通过 `viz_uart_putchar` 拦截所有 xprintf 输出
- **结果**: 虽然消除了文本污染，但 FPS 不变
- **结论**: 文本拦截器工作正常，但不是根因

---

## 3. 根因分析: `uart_write` 阻塞 + Web Toolkit FPS 度量差异

### 3.1 `uart_write` 是阻塞式轮询

`hx_drv_uart.h` 第 869 行的 API 声明:
```c
int32_t (*uart_write)(const void *data, uint32_t len);   /*!< Send data by uart device(blocked) */
```

**`uart_write` 被官方文档标记为 "blocked"（阻塞式）。** 它会等待每个字节物理发送完成后才返回。

### 3.2 精确帧周期计算

**当前 RAW 模式 (16B chunk + 150µs throttle):**
- 每 chunk 含 16B 物理传输 + 150µs 延迟
- 物理传输时间 = 16 × 10bit / 921600bps = 174µs
- 每 chunk 总计 = 174 + 150 = **324µs**
- JPEG 6KB → 375 chunks × 324µs = **~121ms**
- JPEG 1.8KB → 113 chunks × 324µs = **~37ms**

| JPEG 大小 | UART 耗时 | 帧周期 (algo+encode+UART) | 实测 FPS | 理论 FPS |
| :-------- | :-------- | :------------------------ | :------- | :------- |
| 1.8 KB    | ~37ms     | 170 + 15 + 37 = 222ms     | 4.3-4.6  | **4.5**  |
| 6.0 KB    | ~121ms    | 170 + 15 + 121 = 306ms    | 3.9-4.0  | **3.3**  |

> ✅ 实测 FPS 与理论计算吻合。JPEG 越大 → UART 阻塞越久 → FPS 越低，这是物理定律。

### 3.3 ⚠️ 核心发现: Web Toolkit 的 "5.8 FPS" 是误导性指标

Web Toolkit 日志中的 JSON:
```json
{"type": 1, "name": "INVOKE", ..., "algo_tick": [[68160000]], ...}
```

- `algo_tick` = 68,160,000 cycles ÷ 400MHz = **170.4ms**
- 1 / 0.1704 = **5.87 FPS**

**Web Toolkit 前端 JS 直接用 `algo_tick` 计算并显示 FPS，这只是算法帧率，不包含 UART 传输时间。**

Himax 官方的 `send_result.cpp` 同样使用阻塞 `uart_write` (甚至更小的 8 字节 chunk)，所以 Web Toolkit 模式下的 **实际帧交付速率**也远低于 5.8:
- Base64+JSON 体积 ≈ 8200B → UART 物理 ≈ 89ms
- 帧周期 = 170 + 15 + 89 = 274ms → **实际 ≈ 3.6 FPS**

> **结论: RAW 模式 (4.0-4.5 FPS) 实际上已经优于 Web Toolkit (3.6 FPS)，因为 RAW 少了 Base64 膨胀。"5.8 vs 4.0" 的差距根本不存在，是度量指标的误导。**

---

## 4. 实验方案评估与关闭理由

### ❌ 实验 1: UART 时间测量 → 不再需要

**已通过理论计算确认根因。** UART 阻塞时间 = 物理传输时间 + throttle 延迟，完全可预测。无需额外测量。

### ❌ 实验 2: DMA 非阻塞 UART 发送 → 已失败，不值得

| 维度         | 评估                                                                                                 |
| :----------- | :--------------------------------------------------------------------------------------------------- |
| **理论收益** | 帧周期 = max(170ms, 65ms) = 170ms → 5.9 FPS (+48%)                                                   |
| **实际尝试** | 2026-02-26 实施，出现严重问题：内存溢出 (`alloc prev buffer fail`)、DMA 状态判断 bug、代码多处重复行 |
| **内存风险** | WE2 SRAM 仅余 ~26KB，DMA 双缓冲需要额外 buffer，导致 tensor arena 被迫从 1432→1400 KB                |
| **复杂度**   | 需要 cache clean、volatile flag、DMA 完成轮询、双缓冲管理                                            |
| **结论**     | ⛔ **风险/收益比极差。** 4.0 FPS 对光流可视化完全够用，为了 +2 FPS 冒内存崩溃风险不值得               |

> 2026-02-27: 已通过 `git checkout -- .` 回退全部 DMA 修改，恢复到 `f9ec451` 稳定版。光流输出验证正常。

### ❌ 实验 3: 加大 chunk size → 收益可忽略

将 chunk 从 16→1024 字节仅减少函数调用次数 (375→6)，物理传输时间不变 (65ms)。预估节省 ~1-2ms，对 FPS 无可感知影响。

---

## 5. 最终结论

| 指标                     | 值          | 说明                                |
| :----------------------- | :---------- | :---------------------------------- |
| **RAW 实际帧率**         | 4.0-4.5 FPS | 取决于 JPEG 大小                    |
| **Web Toolkit 实际帧率** | ~3.6 FPS    | Base64 膨胀使得 UART 更慢           |
| **Web Toolkit 显示帧率** | 5.8 FPS     | ≠ 实际帧率，是 `algo_tick` 算法帧率 |
| **理论极限 (DMA)**       | 5.9 FPS     | 需要 DMA 异步，但风险过高           |
| **改进空间**             | **无**      | 当前 FPS 已是阻塞 UART 物理极限附近 |

> **🏁 Plan 017 关闭。** RAW 二进制模式 (4.0-4.5 FPS) 已经是当前硬件条件下的最优解。Web Toolkit "5.8 FPS" 是误报算法帧率。无需进一步 UART 优化。

---

## 6. 调试遗留记录 (保留)

### 6.1 文本拦截器 (viz_uart_putchar)
通过劫持 `xdev_out`，在 RAW 模式下丢弃所有 `xprintf` 输出。**已验证有效**。

### 6.2 串口限流器 (保留)
`uart_send_bytes` 中 150µs/16B 延迟**必须保留**，防止 CH340 USB 桥 FIFO 溢出导致 JPEG 损坏。

### 6.3 上位机握手逻辑 (flow_viewer.py)
增加了对 Windows DTR/RTS 重置的抑制，并增加了 `0xFC` 指令的循环握手。**工作正常**。

### 6.4 DMA 实验记录 (历史存档)
- 2026-02-26: 将 `uart_write` 替换为 `uart_write_udma`，遇到 DMA 首次传输卡死、SRAM 溢出
- 2026-02-27: 全部回退 (`git checkout -- .`)，确认光流输出恢复正常
- **教训**: WE2 的 SRAM 余量极小 (~26KB)，任何需要额外 buffer 的 DMA 方案都会触发内存边界问题
