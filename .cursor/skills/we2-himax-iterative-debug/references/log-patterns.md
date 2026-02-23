# Log Pattern Sets

Use these patterns for compact extraction from pipeline logs.

## Core Health Signals

- `initial done`
- `"name": "NAME?"`
- `"name": "MODEL?"`
- `"name": "INVOKE"`
- `[SUMMARY]`
- `[done] pipeline success`

## Camera/Memory Failure Signals

- `WD3_RAW[0]`
- `camera frame capture fail`
- `cv_yolov8n_ob_run fail`
- `wait first camera frame timeout`
- `wait new camera frame timeout`

## JPEG Path Signals

- `viz skip invalid jpeg`
- `cisdp=.../0`
- `sig=` values not matching JPEG SOI expectations

## Decision Hints

1. `NAME?` + `MODEL?` present, but no `INVOKE`: inspect publish gating path.
2. `INVOKE` present, but HTML preview empty: inspect Windows serial ownership and parser path.
3. `WD3_RAW[0]`: prioritize memory/resolution or buffer layout changes.
4. `cisdp size = 0` with repeated JPEG skip: prioritize real JPEG source path.

## Minimal Evidence Bundle

For each run, keep only:

- one extracted key log file
- one summary count file
- one plan/history markdown entry with conclusions
