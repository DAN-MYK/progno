#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# Refresh features.py from training source before building (skip if already symlinked)
cp --no-clobber ../training/src/progno_train/features.py features.py 2>/dev/null || true
uv run pyinstaller \
    --onedir \
    --name progno-sidecar \
    --hidden-import catboost \
    --add-data "features.py:." \
    server.py
# Update Tauri's externalBin lookup (app/sidecar/dist/ = what src-tauri/../sidecar/dist/ resolves to).
# PyInstaller onedir needs the binary AND _internal/ to be adjacent at the call site.
ln -sf ../../../sidecar/dist/progno-sidecar/progno-sidecar \
    ../app/sidecar/dist/progno-sidecar-x86_64-unknown-linux-gnu
ln -sf ../../../sidecar/dist/progno-sidecar/_internal \
    ../app/sidecar/dist/_internal
echo "Built: dist/progno-sidecar/progno-sidecar"
