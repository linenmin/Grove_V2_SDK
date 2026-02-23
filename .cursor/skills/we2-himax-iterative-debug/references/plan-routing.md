# Plan Routing Rules

## Goal

Keep debug memory durable without bloating active context.

## Routing Policy

1. Use the latest plan markdown for short runs likely to finish in a few iterations.
2. Create a dedicated debug history markdown when:
   - more than 3 iterations are expected, or
   - the issue spans multiple sessions/days, or
   - evidence volume is large.
3. Append every new run to that dedicated history file.
4. After success, write only key successful attempts back to the latest plan markdown.
5. Keep full run chronology in the history file; avoid duplicating long logs in the latest plan.

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
