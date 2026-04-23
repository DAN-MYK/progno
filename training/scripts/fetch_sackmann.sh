#!/usr/bin/env bash
set -euo pipefail

# Clone or update Sackmann's tennis_atp repo into data/raw/
# Usage: bash scripts/fetch_sackmann.sh

TARGET="data/raw/tennis_atp"

if [ ! -d "$TARGET" ]; then
    mkdir -p "$(dirname "$TARGET")"
    git clone https://github.com/JeffSackmann/tennis_atp.git "$TARGET"
else
    (cd "$TARGET" && git pull --ff-only)
fi

# Symlink match CSVs to the flat location our ingester expects
cd data/raw
rm -f atp_matches_*.csv
for f in tennis_atp/atp_matches_[0-9]*.csv; do
    ln -sf "$f" "$(basename "$f")"
done

echo "Sackmann data available in $TARGET"
