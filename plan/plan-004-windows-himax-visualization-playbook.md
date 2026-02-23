# Windows + Himax 可视化实操手册（optical_cam_oflow）

## 0. 先说结论（为什么你现在 connect 成功但没输出）

你现在的现象不是单一“串口在 WSL”问题，而是两层叠加：

1. **串口归属问题**  
   - 如果设备 attach 在 WSL，Windows Edge 的 WebSerial 看不到 COM（或拿不到权限）。

2. **协议/通道不匹配问题（核心）**  
   - 现阶段 `optical_cam_oflow` 的 M2 代码在发 `JPEG + META`，但走的是 `spi_master_protocol_write_sp(...)` 数据通道。  
   - Himax HTML（Web toolkit）常见可视化链路是 UART JSON（`send_result.cpp` / `event_reply(...)`）格式。  
   - 所以可能出现：页面已 connect，但一直 loading（没有收到它能解析的数据）。

---

## 1. 你现在该怎么做（推荐流程）

推荐采用“串口接力”：

- **步骤 A（WSL）**：烧录并确认程序运行
- **步骤 B（Windows）**：切回 COM，打开 Himax HTML 观察
- **步骤 C（需要继续开发）**：再切回 WSL

同一时刻串口只能被一个端占用。

---

## 2. 步骤 A：在 WSL 烧录运行

### A1. Windows 管理员 PowerShell：挂到 WSL

```powershell
usbipd list
usbipd attach --wsl Ubuntu-22.04 --busid <BUSID>
```

### A2. WSL：确认串口可用

```bash
ls -l /dev/ttyACM* 2>/dev/null
python3 -c "import serial; s=serial.Serial('/dev/ttyACM0',921600,timeout=1); print('open ok'); s.close()"
```

### A3. 烧录（不改模型，最快）

```bash
bash .cursor/skills/we2-optical-sd-pipeline/scripts/run_optical_pipeline.sh \
  --mode nomodel \
  --app-type optical_cam_oflow \
  --port /dev/ttyACM0 \
  --capture-seconds 10 \
  --keyword "initial done" \
  --keyword "[loop="
```

---

## 3. 模型放在哪里（地址与命令）

当前地址定义：

- 文件：`EPII_CM55M_APP_S/app/scenario_app/optical_cam_oflow/config/common_config.h`
- 宏：`YOLOV8_OBJECT_DETECTION_FLASH_ADDR 0x3AB7B000`

对应 xmodem 的 model offset 通常写为 `0xB7B000`。

示例（改模型时才需要）：

```bash
python3 xmodem/xmodem_send.py \
  --port=/dev/ttyACM0 \
  --baudrate=921600 \
  --protocol=xmodem \
  --file=we2_image_gen_local/output_case1_sec_wlcsp/output.img \
  --model="model_zoo/<your_model>.tflite 0xB7B000 0x00000"
```

如果模型不变，继续 `--mode nomodel` 即可。

---

## 4. 步骤 B：切到 Windows，用 Himax HTML 观察

### B1. 先释放 WSL 串口

- 停掉 WSL 里占用串口的命令（xmodem/串口监控）。

### B2. Windows 管理员 PowerShell：从 WSL 分离

```powershell
usbipd list
usbipd detach --busid <BUSID>
```

### B3. Windows 侧打开 toolkit

1. 下载并解压 `Himax_AI_web_toolkit.zip`  
2. 用 **Microsoft Edge** 打开 `index.html`  
3. 选择 `Grove Vision AI(V2)` -> 点 `Connect` -> 选择 COM3（或你的 COM）

---

## 5. 在 Himax 页面如何判断“有输出”

按从低到高三个层级检查：

1. **连接层**：右上角显示已连接（不是 connect 按钮状态）  
2. **传输层**：页面不再长期停留 `loading...`  
3. **内容层**：预览区域有刷新，或 Output/Console 区出现连续消息

如果一直 loading：

- 先排除串口占用（TeraTerm、WSL、其他串口工具）  
- 再看协议是否匹配（本手册第 0 节）

---

## 6. 步骤 C：看完后切回 WSL 继续开发

### C1. Windows 先断开 HTML 串口

- 页面点 Disconnect 或关闭页面

### C2. Windows 管理员 PowerShell：重新 attach 到 WSL

```powershell
usbipd attach --wsl Ubuntu-22.04 --busid <BUSID>
```

### C3. WSL 复检

```bash
ls -l /dev/ttyACM* 2>/dev/null
python3 -c "import serial; s=serial.Serial('/dev/ttyACM0',921600,timeout=1); print('open ok'); s.close()"
```

---

## 7. 关键提醒（避免反复踩坑）

- 同一时刻只能一个端占用串口（Windows 或 WSL 二选一）。
- `connect 成功` 不等于 `协议匹配`。
- 若要稳定在 Himax HTML 看光流可视化，建议下一步补齐 UART JSON 输出路径（`send_result` / `event_reply` 风格），和当前 SPI meta 输出并存。

