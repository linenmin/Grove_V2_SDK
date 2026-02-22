---
name: discord-notify-wsl2
description: 当用户明确要求「任务完成后用 Discord 通知」时，在 WSL2 环境下通过 DISCORD_WEBHOOK_URL 发送 Discord 消息。使用 bash + curl，临时文件放在 /tmp。
---

# Discord 通知 (WSL2)

在本仓库中，当用户**事先说明**「这次任务完成后帮我用 Discord 通知」时，在任务结束时用环境变量 `DISCORD_WEBHOOK_URL` 向 Discord Webhook 发送一条消息。仅适用于 **WSL2 / Linux** 环境。

---

## 1. 前置条件

- 用户已在 Discord 创建好频道 Webhook，并将 URL 保存在**本地环境变量**中。
- 在 WSL2 中已配置（例如在 `~/.bashrc` 末尾），建议使用 **export** 以便子进程也能读到：
  ```bash
  export DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/..."
  ```
  未加 export 也可以，发送时会用 `bash -i -c` 加载 `.bashrc` 再发。
- 不要将真实 Webhook URL 写进仓库或本 SKILL 文件。

---

## 2. 何时使用

**仅当用户明确要求时**才发送通知，例如：

- 「这次任务完成后帮我用 Discord 通知我」
- 「这个 debugging 做完，用 Discord 提醒一下我」
- 「这次长任务结束后，发一条 Discord 通知」

用户**没有**要求时，不要发送任何消息。

---

## 3. 通知内容

至少包含：

- **任务名称或简短描述**
- **最终状态**：success / failure / partial

可选：耗时、简短详情或下一步建议。

示例（成功）：
```text
✅ Task **nightly-training** finished successfully.
Duration: `0:42:15`
Details: best_val_acc=0.92
```

示例（失败）：
```text
❌ Task **data-cleanup** failed.
Please check Cursor for error details.
```

---

## 4. 发送流程（WSL2：bash + curl）

在 WSL2 下统一使用以下流程，**临时文件放在 `/tmp`**，避免 Windows 路径。

### 步骤

1. **用 write tool** 在 `/tmp/temp_discord_msg.json` 写入 UTF-8 的 JSON，例如：
   ```json
   {"content":"✅ Task **<任务名>** finished successfully.\nDuration: `<时长>`\nDetails: <详情>"}
   ```
   或失败时：
   ```json
   {"content":"❌ Task **<任务名>** failed.\nPlease check Cursor for error details."}
   ```

2. **在终端执行**（不要用 PowerShell）：
   - 使用 `bash -i -c '...'` 启动**交互式** shell，这样会加载 `~/.bashrc`，即使用户在 Cursor 等非交互环境下运行命令，也能读到 `DISCORD_WEBHOOK_URL`（无论是否写了 `export`）。
   ```bash
   bash -i -c 'curl -s -w "\nHTTP_CODE:%{http_code}\n" -X POST -H "Content-Type: application/json" --data-binary @/tmp/temp_discord_msg.json "$DISCORD_WEBHOOK_URL"'
   ```
   - 若出现 `cannot set terminal process group` 等提示，可忽略；只要输出中有 `HTTP_CODE:204`（或 200）即表示发送成功。

3. **用 delete tool** 删除 `/tmp/temp_discord_msg.json`。

4. 若 curl 返回非 2xx（或终端显示 HTTP_CODE 非 204/200），在对话中说明失败原因及 HTTP 状态码。

### 规则

- 消息简短、清晰，不包含敏感信息。
- **始终**使用上述「写文件 → curl → 删文件」流程。
- **始终**发送后删除临时文件。
- **不要**在任何文件或消息中写死完整 Webhook URL。

---

## 5. 环境变量未设置时

若 `$DISCORD_WEBHOOK_URL` 为空，在对话中告知用户，例如：

「未检测到 DISCORD_WEBHOOK_URL。请在 WSL2 的 `~/.bashrc` 末尾添加：  
`export DISCORD_WEBHOOK_URL="你的Webhook URL"`  
然后执行 `source ~/.bashrc` 或重新打开终端。」

不要静默失败。

---

## 6. 手动测试命令（供用户自测）

用户配置好环境变量后，可在终端执行：

```bash
echo "DISCORD_WEBHOOK_URL is set: $( [ -n "$DISCORD_WEBHOOK_URL" ] && echo yes || echo no )"
echo '{"content":"🔔 WSL2 测试：Discord 通知已打通"}' > /tmp/temp_discord_msg.json
curl -s -w "\nHTTP_CODE:%{http_code}\n" -X POST -H "Content-Type: application/json" --data-binary @/tmp/temp_discord_msg.json "$DISCORD_WEBHOOK_URL"
rm -f /tmp/temp_discord_msg.json
```

成功时 Discord 会收到消息，终端显示 `HTTP_CODE:204`（或 200）。
