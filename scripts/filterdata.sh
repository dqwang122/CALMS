#!/usr/bin/env bash
set -e

# bash filterdata.sh raw de -p

DATADIR="$1"
LG="$2"
SPLIT="$3"

if [ -z "$DATADIR" ]; then
    echo "Lose the raw data dir!"
    exit
fi
if [ -z "$LG" ]; then
    echo "Lose the language!"
    exit
fi

SOURCE=(bbc france24 faz)


echo "Clear and Split dataset..."
if [ ! -d "$DATADIR/clean"  ];then
  mkdir "$DATADIR/clean"
fi
if [ -f "$DATADIR/clean/$LG.jsonl"  ];then
  echo "Remove existing $DATADIR/clean/$LG.jsonl"
  rm "$DATADIR/clean/$LG.jsonl"
fi

for SRC in ${SOURCE[*]}
  do
    python -m preprocess.dataClear -i "$DATADIR/$SRC/$LG/data.txt" -o "$DATADIR/clean/$LG.jsonl" -l "$LG" -t 0.8 -c
  done