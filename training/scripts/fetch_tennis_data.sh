#!/usr/bin/env bash
# Download tennis-data.co.uk XLSX files for recent years not yet in Sackmann.
# Saves to training/data/raw/tennis_data_xlsx/{atp,wta}_{year}.xlsx
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="$SCRIPT_DIR/../data/raw/tennis_data_xlsx"
mkdir -p "$OUT_DIR"

BASE="https://www.tennis-data.co.uk"
START_YEAR="${1:-2025}"
END_YEAR="$(date +%Y)"

_download() {
    local url="$1"
    local out="$2"
    if [ -f "$out" ] && [ "$(stat -c%s "$out" 2>/dev/null || echo 0)" -gt 10000 ]; then
        echo "  already have $out ($(stat -c%s "$out") bytes), skipping"
        return 0
    fi
    echo "  fetching $url"
    # Try normal TLS first, fall back to --insecure for sites with old cert chains
    if curl -fsSL --max-time 30 -o "$out" "$url" 2>/dev/null; then
        echo "  saved $out ($(stat -c%s "$out") bytes)"
    elif curl -fsSL --max-time 30 --insecure -o "$out" "$url" 2>/dev/null; then
        echo "  saved $out via --insecure ($(stat -c%s "$out") bytes)"
    else
        echo "  WARN: could not download $url — skipping"
        rm -f "$out"
    fi
}

echo "[fetch_tennis_data] downloading years $START_YEAR–$END_YEAR"
for year in $(seq "$START_YEAR" "$END_YEAR"); do
    echo "=== $year ==="
    _download "${BASE}/${year}/${year}.xlsx"   "$OUT_DIR/atp_${year}.xlsx"
    _download "${BASE}/${year}w/${year}w.xlsx" "$OUT_DIR/wta_${year}.xlsx"
done

echo "[fetch_tennis_data] done. files in $OUT_DIR:"
ls -lh "$OUT_DIR" 2>/dev/null || echo "  (empty)"
