#!/usr/bin/env bash
set -euo pipefail

WTA_DIR="training/data/raw/tennis_wta"

if [ -d "$WTA_DIR/.git" ]; then
    echo "Pulling latest tennis_wta..."
    git -C "$WTA_DIR" pull --ff-only
else
    echo "Cloning tennis_wta..."
    git clone https://github.com/JeffSackmann/tennis_wta "$WTA_DIR"
fi

echo "WTA data ready at $WTA_DIR"
