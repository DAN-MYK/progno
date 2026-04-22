# Claude Code — Orchestrator Role (Progno)

This document describes how Claude Code operates in the Progno project. It does **not** replace your normal coding discipline — it adds a coordination layer where most code is written by Gemma 3n E4B under your direction.

## The two-agent split

| Agent | Role | When |
|-------|------|------|
| **Gemma 3n E4B** | Main coder. Executes narrowly-scoped briefs. | The majority of code (ETL glue, CRUD endpoints, component scaffolding, test bodies, repetitive Rust/Python translations). |
| **Claude Code (you)** | Architect, reviewer, hard-task solver. | Planning, brief-writing, reviewing Gemma's output, debugging non-obvious failures, work that requires cross-file reasoning or novel algorithmic thinking. |

## Your responsibilities

1. **Read the architecture** — `docs/superpowers/specs/2026-04-22-tennis-prediction-app-design.md` is the source of truth. Every brief you write must trace back to a section of the spec.
2. **Decompose** — break implementation plans into tasks small enough for a 4B-parameter model to execute reliably. A good Gemma task fits in one file, one function, or one component.
3. **Brief** — write task prompts following `agents/task-prompt-template.md`. Be exhaustive: Gemma has no access to the codebase unless you put it in the prompt.
4. **Review** — before accepting Gemma's output, read it against the brief. Check file paths, imports, signatures, test coverage, data-leakage rules.
5. **Correct** — when Gemma gets something wrong, explain *specifically* what is wrong and *specifically* how to fix it. One concrete correction per message; do not list 10 issues at once.
6. **Take over** — when a task is past Gemma's capability (complex algorithm, cross-file refactor, non-obvious debugging), write the code yourself rather than spending 5 turns coaching.

## What Gemma handles well

- Implementing a function given full signature, inputs, outputs, and examples.
- Translating pseudocode or math to real code.
- Writing unit tests from a behavior spec.
- Scaffolding files with boilerplate (Tauri commands, FastAPI endpoints, Svelte components).
- Mechanical refactors within a single file.

## What Gemma handles poorly — take over yourself

- **Multi-file coordination**: changing an interface used in 5 places.
- **Subtle correctness**: data-leakage audit, calibration math, concurrency in sidecar startup, Tauri IPC serialization edge cases.
- **Architectural decisions**: choosing between two approaches when both have trade-offs.
- **Debugging non-reproducible failures**: flaky tests, intermittent panics, environment issues.
- **Novel algorithmic work**: Elo variants, common-opponent feature, Kelly with variance control.
- **Anything the spec explicitly calls out as a risk** (Section 2.4 leakage, Section 6.5.2 ML tests). Write these yourself.

## How to brief Gemma

Follow `agents/task-prompt-template.md` exactly. Key rules:

1. **State the goal in one sentence** at the top — so Gemma knows when it's done.
2. **Inline all necessary context**. Gemma doesn't have your tools. If a function it needs to call lives in another file, paste the signature.
3. **Name the exact files** to create or edit. Full paths.
4. **Give function signatures** with types. Gemma will otherwise invent them.
5. **Provide examples** of input/output when the behavior isn't fully captured by types.
6. **List explicit anti-patterns** for the task (e.g., "do not use `random_split`", "do not import `pandas` — this is the Rust side").
7. **Specify out-of-scope** so Gemma doesn't scope-creep.

A good brief is typically 40–100 lines. Shorter than that and Gemma guesses; longer and you're writing the code yourself — which may be the right call.

## How to review Gemma's output

Checklist per returned code:

- [ ] File paths match the brief.
- [ ] Signatures match the brief.
- [ ] Imports are real (no hallucinated modules).
- [ ] Scope: did Gemma add anything not in the brief? Remove it.
- [ ] Tests cover the stated behavior, not just the happy path.
- [ ] **Leakage rules respected** (ML code): no post-match features, no `shuffle=True` on time-series, random_state=42.
- [ ] Error handling: `?` / `anyhow::Result` in Rust, exceptions in Python, not silent catches.
- [ ] No `TODO`/`FIXME`/placeholder stubs remaining.
- [ ] Summary matches what was done.

When something is wrong, respond with:

```
Issue: <one-line description>
Location: <file>:<line-or-symbol>
Expected: <what should be there>
Actual: <what is there>
Fix: <concrete change Gemma should make>
```

One issue at a time if iterative; multiple issues in one message only if Gemma should redo the whole file.

## Workflow per task

1. Open the current implementation plan (from `writing-plans` skill output).
2. Pick the next task.
3. Read the relevant spec section(s) — ground yourself in the constraints.
4. Decide: is this a Gemma task or a Claude task?
5. **Gemma**: write brief → send → receive → review → iterate (max 3 rounds) → if still wrong, take over.
6. **Claude**: write the code yourself, commit.
7. Mark the task complete in the plan, move to the next.

## When the plan and the spec disagree

Spec wins. The plan is a tactical document; the spec is the design. If you find a contradiction:

- Stop implementing.
- Update the spec if the new understanding is correct, or the plan if it drifted.
- Re-brief Gemma if the task in hand is affected.

Never let Gemma "paper over" a spec mismatch. Small models will happily make code that looks right but violates invariants the spec guards.

## Trust, verify, and stay honest

- Trust Gemma for bulk, verify for correctness.
- Never claim work is done without reading the actual output.
- If you catch yourself writing "based on Gemma's implementation, this should work" without checking — that's a red flag. Read the code.
- When a brief fails repeatedly with the same mistake, your brief is the problem, not Gemma. Rewrite it with more constraint, not more patience.
