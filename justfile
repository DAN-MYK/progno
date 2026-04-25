default:
    @just --list

# --- Phase 1a targets ---
update-data:
    bash training/scripts/fetch_sackmann.sh
    bash training/scripts/fetch_sackmann_wta.sh

ingest:
    cd training && uv run python -m progno_train.cli --tour atp ingest

elo:
    cd training && uv run python -m progno_train.cli --tour atp elo

publish version:
    cd training && uv run python -m progno_train.cli --tour atp publish {{version}}

# --- Phase 3 targets (ATP) ---
features:
    cd training && uv run python -m progno_train.cli --tour atp features

train:
    cd training && uv run python -m progno_train.cli --tour atp train

validate:
    cd training && uv run python -m progno_train.cli --tour atp validate

retrain version:
    cd training && uv run python -m progno_train.cli --tour atp retrain --version {{version}}

build-sidecar:
    cd sidecar && bash build.sh

# --- Phase 4 targets (WTA) ---
ingest-wta:
    cd training && uv run python -m progno_train.cli --tour wta ingest

elo-wta:
    cd training && uv run python -m progno_train.cli --tour wta elo

features-wta:
    cd training && uv run python -m progno_train.cli --tour wta features

train-wta:
    cd training && uv run python -m progno_train.cli --tour wta train

validate-wta:
    cd training && uv run python -m progno_train.cli --tour wta validate

retrain-wta version:
    cd training && uv run python -m progno_train.cli --tour wta retrain --version {{version}}

# --- Dev helpers ---
test:
    cd training && uv run pytest -v

test-rust:
    cd app/src-tauri && cargo test

test-all: test test-rust

fmt:
    cd training && uv run ruff format .
    cd training && uv run ruff check --fix .

check:
    cd training && uv run ruff check .
    cd training && uv run pytest -v

# --- Phase 3.5 targets ---
# Run `just fetch-odds-data` first to download XLSX files, then these targets
# re-run the elo step which joins odds when XLSX files are present (Phase 3.5 Task 4).
fetch-odds-data:
    bash training/scripts/fetch_tennis_data.sh

ingest-odds:
    cd training && uv run python -m progno_train.cli --tour atp elo
    cd training && uv run python -m progno_train.cli --tour wta elo

ingest-odds-atp:
    cd training && uv run python -m progno_train.cli --tour atp elo

ingest-odds-wta:
    cd training && uv run python -m progno_train.cli --tour wta elo
