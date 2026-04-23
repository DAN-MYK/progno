default:
    @just --list

# --- Phase 1a targets ---
update-data:
    cd training && uv run python -m progno_train.cli update_data

ingest:
    cd training && uv run python -m progno_train.cli ingest

elo:
    cd training && uv run python -m progno_train.cli elo

publish version:
    cd training && uv run python -m progno_train.cli publish {{version}}

# --- Phase 3 targets ---
features:
    cd training && uv run python -m progno_train.cli features

train:
    cd training && uv run python -m progno_train.cli train

validate:
    cd training && uv run python -m progno_train.cli validate

retrain version:
    cd training && uv run python -m progno_train.cli retrain --version {{version}}

build-sidecar:
    cd sidecar && uv run pyinstaller --onefile --name progno-sidecar server.py

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
