#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
uv run pyinstaller \
  --onefile \
  --name progno-sidecar \
  --hidden-import catboost \
  --hidden-import progno_train.features \
  server.py
echo "Built: dist/progno-sidecar"
