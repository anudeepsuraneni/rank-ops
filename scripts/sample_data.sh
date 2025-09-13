#!/usr/bin/env bash
set -e

RAW="data/raw/ml-25m/ratings.csv"
SAMPLED="data/sample/ml-25m/ratings_50k.csv"

if [ ! -f "$SAMPLED" ]; then
  echo "Creating sampled ratings (50k rows) for CI..."
  head -n 50001 "$RAW" > "$SAMPLED"
fi
