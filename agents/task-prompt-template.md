# Task Prompt Template — Claude → Gemma

Copy this structure for every task you brief to Gemma. Fill in every section; omit only `Examples` if the types and one-line description already fully determine behavior (rare). Keep briefs between 40 and 100 lines when possible.

---

```
# Task: <3–7 word name of what's being built>

## Goal
<One sentence. What must be true when you're done.>

## Spec reference
<Section number(s) in docs/superpowers/specs/2026-04-22-tennis-prediction-app-design.md that this implements. Summarize the relevant constraint in 1-2 lines so Gemma doesn't have to open the file.>

## Files
<Exact paths. State whether each is new or existing. For existing: paste the current relevant content inline below under "Existing code to work with", so Gemma has the context.>

- CREATE: training/src/progno_train/elo.py
- EDIT:   training/src/progno_train/pipelines/ingest.py (add one function, do not touch the rest)

## Existing code to work with
<Paste any current code Gemma needs to integrate with. Function signatures, type definitions, data schemas. If Gemma needs the shape of a DataFrame, describe the columns + types; do not assume Gemma can infer.>

## Requirements

- <Bullet list of what the code must do. Ordered by importance.>
- <Each bullet should be directly verifiable from the code or a test.>
- <Avoid "should be efficient", "should be clean" — those are unverifiable. Say "must run in <1s for 10k rows" or "must not allocate in the hot loop".>

## Function signatures

```python
# Provide the exact signatures. Types included.
def compute_elo_update(
    rating_a: float,
    rating_b: float,
    winner: Literal["A", "B"],
    k_factor: float,
) -> tuple[float, float]:
    ...
```

## Examples

<Input/output pairs that demonstrate the expected behavior. At least one happy path, at least one edge case.>

Input:
    rating_a=1500, rating_b=1500, winner="A", k_factor=32
Expected output:
    (1516.0, 1484.0)

Input:
    rating_a=1800, rating_b=1500, winner="B", k_factor=32
Expected output:
    (1775.2, 1524.8)  # approximate; assert within ±0.1

## Tests

<Name the tests to write. Give the assertion content, not the implementation.>

- test_equal_ratings_symmetric_update: equal ratings → ±k/2 delta
- test_upset_favors_underdog_more: lower-rated winner gains more than higher-rated winner does
- test_k_zero_means_no_change: k=0 → ratings unchanged

## Anti-patterns (do not)

- Do not use numpy for this — pure Python is fast enough.
- Do not add a global state for ratings; this function is pure.
- Do not cache results with lru_cache — values are floats, not hashable safely.
- Do not handle winner=None or winner="Draw" — tennis has no draws, brief does not require it.

## Out of scope

- The state management for a player's rating over time (separate task).
- Surface-specific logic (separate task).
- K-factor calculation (you receive it as input; another task computes it).

## Done when

- All tests pass (`pytest training/tests/test_elo.py -v`).
- `ruff check` and `ruff format` pass.
- No imports outside stdlib + already-used project deps.
```

---

## Template notes (for Claude, not sent to Gemma)

- **Spec reference** keeps you honest. Every task should map to the spec, not be invented.
- **Existing code to work with** is the section most often skipped, which is why Gemma halucinates the most often. Be generous here.
- **Requirements** are verifiable assertions. "Must be clean" fails. "Must complete in under 500ms on 10k-row input" passes.
- **Anti-patterns** are where you pre-empt specific Gemma mistakes you've seen. Keep a running note of which anti-patterns recur and always include them.
- **Done when** is the contract. Without it, Gemma decides, and Gemma's bar is lower than yours.

## Tuning knobs for hard briefs

When a brief fails twice in a row, tighten these:

1. Shrink scope. Split into two briefs.
2. Add a reference implementation ("look at `elo_update` in line 45 of `../elo.py` for the style I want").
3. Add explicit type hints everywhere, even in examples.
4. Add one more anti-pattern matching the mistake Gemma made.

If a brief fails three times, take over yourself. The brief design cost is already higher than the implementation cost.
