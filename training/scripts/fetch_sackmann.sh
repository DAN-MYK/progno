#!/usr/bin/env bash
set -euo pipefail

# Clone or update Sackmann's tennis_atp repo into training/data/raw/
# Usage: bash training/scripts/fetch_sackmann.sh  (from repo root)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="$SCRIPT_DIR/../data/raw/tennis_atp"

if [ ! -d "$TARGET" ]; then
    mkdir -p "$(dirname "$TARGET")"
    git clone https://github.com/JeffSackmann/tennis_atp.git "$TARGET"
else
    (cd "$TARGET" && git pull --ff-only)
fi

echo "Sackmann ATP data available in $TARGET"
