# Plan Routing Rules

## Goal

Keep debug memory durable without bloating active context.

## Context Entry Policy

Before deciding where to write, always bootstrap with:

1. `plan/plan-000-context-index.md`
2. `logs/context/context_snapshot_latest.md`
3. Latest debug plan pointed by the snapshot (or latest `plan-00x` non-index file)

Do not read all legacy plans by default. Backtrack only when:

- current runtime evidence conflicts with prior conclusions
- required file/macro provenance cannot be resolved from current plan + snapshot
- a regression suggests a previously reverted change

## Routing Policy

1. Use the latest plan markdown for short runs likely to finish in a few iterations.
2. Create a dedicated debug history markdown when:
   - more than 3 iterations are expected, or
   - the issue spans multiple sessions/days, or
   - evidence volume is large.
3. Append every new run to that dedicated history file.
4. After success, write only key successful attempts back to the latest plan markdown.
5. Keep full run chronology in the history file; avoid duplicating long logs in the latest plan.
6. After every run, refresh snapshot via `bash scripts/build_context_snapshot.sh` before the next attempt.

## Required Fields Per Run

- Hypothesis
- Minimal change
- Exact verify command
- Key evidence lines
- Conclusion and next action

## Anti-Patterns

- Running a new attempt before writing the previous run summary
- Pasting full UART/base64 dumps into plan files
- Mixing multiple hypotheses in one run
- Modifying many files without isolating a single causal change

## Related

Content structure and compaction: `plan-writing.md`.
