# Log Extractor Ablation

- date: 2026-03-13
- experiment_window_start: 2026-03-13T14:35:44+01:00
- experiment_window_end: 2026-03-13T14:37:54+01:00
- experiment_goal: make the child agent more generic while keeping judgment authority in the main agent
- repository_skill_root: `/home/enmin/Seeed_Grove_Vision_AI_Module_V2/.cursor/skills/log-extractor`
- experiment_log: `/home/enmin/Seeed_Grove_Vision_AI_Module_V2/logs/pipeline/pipeline_with-model_optical_cam_oflow_20260313_134701.log`
- experiment_log_shape:
  - total_lines: 200
  - invoke_lines: 28
  - opaque_lines_over_1000_chars: 27
  - avg_invoke_line_length: 5897.3
  - max_line_length: 12401

## Why This Log

This is a real deployment log from the current `144x192` optical-flow mainline. It is a good stress case because it mixes:

- boot and model metadata
- serial runtime statistics
- repeated flow loop lines
- very large single-line `INVOKE` JSON payloads with base64 image data

This combination is exactly where a generic log-extractor either helps the main agent or becomes useless.

## Baseline

- source: `/mnt/d/Dataset/.agents/skills/log-extractor/SKILL.md`
- note: the original draft copies were removed after validation; this report keeps the comparison summary
- baseline_characteristic:
  - good at generic keyword extraction
  - weak boundary against analysis drift
  - too tied to keyword-centric extraction
  - no direct strategy for repeated opaque payload lines

## Evaluation Rubric

Each version is scored manually from 1 to 5 on:

- boundary_purity: does it stop the child agent from judging
- raw_evidence_fidelity: does it preserve original evidence
- coverage_quality: does it guide selection of start, turning points, and end
- compression_quality: does it reduce spam and opaque payload noise
- generality: will it still work outside deployment tasks

Total score is out of 25.

Implementation times are approximate manual edit-and-review times for each increment, not model runtime.

## Variants

### v0 baseline

- file: removed after validation
- change: none
- implementation_time_min: 2
- scores:
  - boundary_purity: 3
  - raw_evidence_fidelity: 4
  - coverage_quality: 2
  - compression_quality: 1
  - generality: 4
  - total: 14
- notes:
  - It already says "do not suggest", but it does not strongly block conclusion leakage.
  - It assumes keyword-centered extraction, which is weak for logs that are noisy but not strictly error-driven.
  - On this deployment log it would likely return giant `INVOKE` lines and waste context.

### v1 boundary only

- file: removed after validation
- change:
  - hardens the role into "extract only, no judgment"
  - removes any ambiguity about success or failure claims
- implementation_time_min: 7
- scores:
  - boundary_purity: 5
  - raw_evidence_fidelity: 4
  - coverage_quality: 2
  - compression_quality: 1
  - generality: 5
  - total: 17
- notes:
  - This is the most important single fix.
  - It solves role drift, but it still does not tell the child agent how to cope with high-noise logs.

### v2 boundary plus signal-based selection

- file: removed after validation
- change:
  - keeps v1 boundary
  - adds start, first-important-state, turning-point, and end selection priorities
  - weakens dependence on fixed error keywords
- implementation_time_min: 9
- scores:
  - boundary_purity: 5
  - raw_evidence_fidelity: 4
  - coverage_quality: 5
  - compression_quality: 2
  - generality: 5
  - total: 21
- notes:
  - On the experiment log this version would more reliably surface `model io`, `initial done`, first `INVOKE`, and late loop statistics.
  - It is much better for generic engineering logs where the most important lines are not tagged as errors.

### v3 boundary plus signal selection plus repeat and payload compression

- file: promoted to the current default `SKILL.md`, temporary duplicate removed after validation
- change:
  - keeps v2 improvements
  - adds explicit handling for repeated lines
  - adds explicit compression of opaque payloads while preserving the shell of the original line
- implementation_time_min: 11
- scores:
  - boundary_purity: 5
  - raw_evidence_fidelity: 4
  - coverage_quality: 5
  - compression_quality: 5
  - generality: 5
  - total: 24
- notes:
  - This version best matches the actual deployment log, where 27 lines exceed 1000 characters and most of that length is base64 image payload.
  - Without payload compression, the child agent would still return too much noise even if its judgment boundary were correct.
  - This is the best version for the repo default skill.

## Chosen Default

The repo default skill is `v3`, written to:

- `/home/enmin/Seeed_Grove_Vision_AI_Module_V2/.cursor/skills/log-extractor/SKILL.md`

Reason:

- v1 fixes the most important role problem
- v2 makes the skill useful on non-error-driven logs
- v3 is the first version that materially solves the selected deployment log's actual context-bloat problem

## Cross Validation

Second validation log:

- `/home/enmin/Seeed_Grove_Vision_AI_Module_V2/logs/pipeline/pipeline_with-model_optical_cam_oflow_20260313_125655.log`

Shape:

- total_lines: 189
- invoke_lines: 26
- opaque_lines_over_1000_chars: 25
- max_line_length: 8921

Key difference from the first log:

- model IO is `in(h=150,w=200,c=6) out(h=160,w=208,c=2)`
- `INVOKE` resolution is repeatedly `[320, 240]`, which corresponds to the fallback camera path rather than the expected `144x192` flow path

Cross-validation result:

- `v1` still fixes the role boundary, but remains too weak against huge `INVOKE` lines
- `v2` would likely surface `model io`, `initial done`, first `INVOKE`, and the repeated `320x240` signal, so it is materially better than `v1`
- `v3` remains best because this second log still contains large opaque payloads on almost every `INVOKE` line, so explicit payload compression is still necessary

Conclusion after cross-validation:

- `v3` remains the best default, not just for the successful `144x192` deployment log, but also for a different deployment state with fallback behavior

## Recommended Reading Order

For future editing of this skill:

1. read the current default `SKILL.md`
2. read this ablation note
3. compare against the original external source if needed

## Limits Of This Experiment

- Quality scoring is manual, not model-benchmarked
- Only one real project log was used
- Cross-validation added one more real deployment log with a different runtime state, but this is still deployment-heavy
- Future non-deployment tasks should still be spot-checked against the same rubric

## Cleanup

The temporary draft variants used during ablation were removed after validation.

Kept:

- current default skill: `/home/enmin/Seeed_Grove_Vision_AI_Module_V2/.cursor/skills/log-extractor/SKILL.md`
- this experiment note
