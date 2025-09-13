#!/usr/bin/env bash
set -e

DATA_DIR="data/raw/ml-25m"
mkdir -p $DATA_DIR

URL="https://files.grouplens.org/datasets/movielens/ml-25m.zip"
ZIP_FILE="data/raw/ml-25m.zip"

if [ ! -f "$ZIP_FILE" ]; then
  echo "Downloading MovieLens 25M..."
  curl -L $URL -o $ZIP_FILE
fi

echo "Extracting..."
unzip -o $ZIP_FILE -d data/raw
