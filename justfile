default:
    @just --list

# --- Phase 1a targets ---
update-data:
    bash training/scripts/fetch_sackmann.sh
    bash training/scripts/fetch_sackmann_wta.sh
    bash training/scripts/fetch_tennis_data.sh

fetch-new-data:
    bash training/scripts/fetch_sackmann.sh
    bash training/scripts/fetch_sackmann_wta.sh
    bash training/scripts/fetch_tennis_data.sh

# Full retrain both tours from scratch (ingest → elo → features → train → validate)
retrain-all:
    just ingest && just elo && just features && just train && just validate
    just ingest-wta && just elo-wta && just features-wta && just train-wta && just validate-wta

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
