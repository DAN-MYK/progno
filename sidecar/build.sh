#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# Refresh features.py from training source before building
cp ../training/src/progno_train/features.py features.py
uv run pyinstaller \
    --onefile \
    --name progno-sidecar \
    --hidden-import catboost \
    --add-data "features.py:." \
    server.py
echo "Built: dist/progno-sidecar"
