#!/usr/bin/env bash
# Download tennis-data.co.uk XLSX files for ATP and WTA.
# Usage: bash training/scripts/fetch_tennis_data.sh [start_year] [end_year]
# Defaults: ATP from 2000, WTA from 2007, up to current year.
set -euo pipefail

OUTDIR="$(dirname "$0")/../data/raw/tennis_data_xlsx"
mkdir -p "$OUTDIR"

CURRENT_YEAR=$(date +%Y)
ATP_START="${1:-2000}"
WTA_START="${2:-2007}"
END_YEAR="${3:-$CURRENT_YEAR}"

echo "Downloading ATP XLSX: $ATP_START–$END_YEAR"
for year in $(seq "$ATP_START" "$END_YEAR"); do
    dest="$OUTDIR/atp_${year}.xlsx"
    if [ -f "$dest" ]; then
        echo "  skip $dest (exists)"
        continue
    fi
    url="http://www.tennis-data.co.uk/${year}/${year}.xlsx"
    echo "  fetch $url"
    curl -sS -L --fail --retry 3 -o "$dest" "$url" || { echo "  WARN: $url not found"; rm -f "$dest"; }
done

echo "Downloading WTA XLSX: $WTA_START–$END_YEAR"
for year in $(seq "$WTA_START" "$END_YEAR"); do
    dest="$OUTDIR/wta_${year}.xlsx"
    if [ -f "$dest" ]; then
        echo "  skip $dest (exists)"
        continue
    fi
    url="http://www.tennis-data.co.uk/${year}w/${year}w.xlsx"
    echo "  fetch $url"
    curl -sS -L --fail --retry 3 -o "$dest" "$url" || { echo "  WARN: $url not found"; rm -f "$dest"; }
done

echo "Done. Files in $OUTDIR:"
ls "$OUTDIR" | head -20
