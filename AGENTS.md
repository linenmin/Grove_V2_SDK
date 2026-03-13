# AGENTS

<skills_system priority="1">

## Available Skills

## Project Entry For 0-Context Agents

Before doing substantive work on this repo, read these files in order:

1. `plan/README.md`
2. `docs/MINIMAL_DEPLOYMENT.md`
3. `docs/DEPLOYMENT.md`
4. `docs/MODEL_EXPORT.md`
5. `docs/KNOWLEDGE_BASE.md`
6. `plan/plan-018-optical-flow-project-reorganization.md`

Current project baseline:

- Treat `157x203 -> 160x208` as the current valid deployment baseline.
- Treat `158x202` and above as runtime-budget experiments, not the default route.
- Treat `144x192` and `150x200 -> 160x208` as older validated/experimental history, not the default route.
- Do not rewrite the root `README.md`; keep the upstream Seeed semantics intact.

If you only need the fastest reproducible route, use `docs/MINIMAL_DEPLOYMENT.md` first.
If the model file is missing or needs regeneration, use `docs/MODEL_EXPORT.md`.

<!-- SKILLS_TABLE_START -->
<usage>
When users ask you to perform tasks, check if any of the available skills below can help complete the task more effectively. Skills provide specialized capabilities and domain knowledge.

How to use skills:
- Invoke: `npx openskills read <skill-name>` (run in your shell)
  - For multiple: `npx openskills read skill-one,skill-two`
- The skill content will load with detailed instructions on how to complete the task
- Base directory provided in output for resolving bundled resources (references/, scripts/, assets/)

Usage notes:
- Only use skills listed in <available_skills> below
- Do not invoke a skill that is already loaded in your context
- Each skill invocation is stateless
</usage>

<available_skills>

<skill>
<name>discord-notify-wsl2</name>
<description>当用户明确要求「任务完成后用 Discord 通知」时，在 WSL2 环境下通过 DISCORD_WEBHOOK_URL 发送 Discord 消息。使用 bash + curl，临时文件放在 /tmp。</description>
<location>project</location>
</skill>

<skill>
<name>project-governance</name>
<description>项目治理与文档更新协议 (Index & Knowledge Base Update Policy). Use when creating new plans or after major technical milestones to ensure plan-000 and KNOWLEDGE_BASE are in sync.</description>
<location>project</location>
</skill>

<skill>
<name>we2-himax-iterative-debug</name>
<description>Iterative debugging playbook for Grove Vision AI V2 (WE2) and Himax AI Web Toolkit across WSL2 and Windows serial handoff. Use when tasks involve build/flash/UART verification, HTML preview issues, compact log extraction, and incremental debug history updates across changing plan markdown files.</description>
<location>project</location>
</skill>

<skill>
<name>we2-optical-sd-pipeline</name>
<description>Runs the Grove Vision AI V2 optical_sd/optical_cam_oflow firmware iteration pipeline in WSL2 with dual flash modes (nomodel and with-model), including build, image generation, xmodem flash, UART keyword verification, and USB re-attach guidance. Supports agent-visible visualization: --viz-camera + --extract-frames to extract INVOKE images for agent to read. Use when user mentions optical_sd, optical_cam_oflow, cvapp_optical_flow.cpp, flash_img_opticalSD, xmodem, usbipd, agent 可见, 可视化调试, extract-frames, pipeline.</description>
<location>project</location>
</skill>

<skill>
<name>windows-observation-workflow</name>
<description>Windows 观察 vs WSL2 调试的流程区分. Use when the task involves visualization on Windows or serial handoff for Himax AI Web Toolkit.</description>
<location>project</location>
</skill>

</available_skills>
<!-- SKILLS_TABLE_END -->

</skills_system>
