# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Progno — AI-Assisted Development

**Progno** is a personal desktop app (Tauri 2 + Rust + Svelte 5 + Python sidecar) that predicts ATP tennis match winners pre-match and helps find value bets by comparing model probability against bookmaker odds.

## Source of truth

- **Architecture spec**: `docs/superpowers/specs/2026-04-22-tennis-prediction-app-design.md`. Every implementation decision traces back to a section of this document. If the spec and the code drift, update the spec or fix the code — never let them disagree silently.

## Development workflow

Claude Code writes all code. Before touching any file:

1. Read `docs/superpowers/specs/2026-04-22-tennis-prediction-app-design.md` — every decision traces back to a section.
2. Check `docs/superpowers/plans/` for the active phase plan and pick the next task.
3. When spec and plan disagree, spec wins — update the plan, not the code.

## Current status

- **Active phase**: Phase 1a — Python ETL pipeline (`training/`)
- **Implementation plan**: `docs/superpowers/plans/2026-04-22-phase-1a-training-pipeline.md`
- `training/` and `app/` are populated as phases are implemented; neither exists yet at project start

## Commands

All commands assume `just` is available. The `training/` workspace uses `uv` for dependency management.

```sh
# From training/
uv run pytest                  # run all tests
uv run pytest tests/test_elo.py  # run a single test file
uv run ruff format .           # format code
uv run ruff check .            # lint

# Via justfile (once created):
just check                     # ruff + pytest
just ingest                    # ingest Sackmann CSVs → parquet
just elo                       # compute Elo → artifacts
just retrain                   # full pipeline (ingest → features → train → validate → publish)
```

## Tech stack & conventions

| Layer | Tech | Key rules |
|-------|------|-----------|
| Python (training) | Python 3.12+, `uv`, pandas, pyarrow, pytest, ruff | Type hints always; `ruff format`; no `random_split` on time-series |
| Rust (app backend) | Rust 2021, Tauri 2, `anyhow::Result` | Idiomatic; inline tests; no `unwrap()` on user-facing paths |
| Frontend | Svelte 5 runes syntax, TypeScript, Tailwind | No class-based components; runes only |

## Phases (from spec §1)

Each phase is independently shippable. Implementation proceeds in order:

1. Tauri skeleton + paste parser + Elo (Rust)
2. + EV / fractional Kelly (Rust)
3. + Python ETL + CatBoost + Platt calibration + sidecar
4. + WTA model
5. + injury toggle, retrain-from-UI, optional schedule scraper

Future improvements (spec §7) — including LLM/news integration via OpenRouter — are out of scope until Phase 5 is stable.

## Non-negotiable invariants

These come from the spec and must be preserved in all code:

- **No data leakage** in ML features (spec §2.4). Post-match stats are never features. Time windows are strictly `[match_date - N, match_date)`.
- **Walk-forward validation only** on historical data. Never random-split time-series.
- **Deterministic training** with `random_seed=42` unless a brief explicitly says otherwise.
- **Kelly stake** is `max(0, 0.25 × full_kelly × bankroll)`. No full Kelly, no negative stakes.
- **Calibration gate** (spec §6.4): model does not get published if it fails log-loss, ECE, or ROI thresholds.
- **Acceptance gate is a hard gate** — the orchestrator never routes around it.

## Repo layout

```
progno/
├── CLAUDE.md                                       # this file
├── docs/superpowers/specs/
│   └── 2026-04-22-tennis-prediction-app-design.md  # architecture spec
├── docs/superpowers/plans/                          # per-phase implementation plans
├── agents/                                          # workflow docs (historical, may be stale)
├── justfile                                        # dev commands (see spec §6.2)
├── training/                                       # Python: ETL, features, training
│   ├── src/progno_train/                           # package source
│   ├── tests/                                      # pytest suite
│   ├── data/raw/                                   # Sackmann CSVs + tennis-data.co.uk XLSX
│   ├── data/staging/                               # intermediate parquets
│   ├── artifacts/                                  # elo_state.json, players.parquet, match_history.parquet
│   └── scripts/fetch_sackmann.sh                   # downloads Sackmann data
└── app/                                            # Tauri: Rust backend + Svelte frontend
    ├── src-tauri/src/                              # Rust (elo.rs, parser.rs, kelly.rs, …)
    └── src/                                        # Svelte 5 components
```
