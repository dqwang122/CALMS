#!/usr/bin/env bash
set -e

# bash translead.sh train.en.lead2 en

DATA="$1"
SRCLG="$2"

LG="de en fr zh"

echo "Split ${DATA} to 5 chunked file..."
split ${DATA} -n l/5 -d ${DATA}.chunk_

echo "Translate chunked file..."
python translate.py --input ${DATA}.chunk_00 --output $DATA.chunk_00.de --src_lang $SRCLG --trg_lang de --batch_size 32
python translate.py --input ${DATA}.chunk_01 --output $DATA.chunk_01.en --src_lang $SRCLG --trg_lang en --batch_size 32
python translate.py --input ${DATA}.chunk_02 --output $DATA.chunk_02.fr --src_lang $SRCLG --trg_lang fr --batch_size 32
python translate.py --input ${DATA}.chunk_03 --output $DATA.chunk_03.zh --src_lang $SRCLG --trg_lang zh --batch_size 32
python translate.py --input ${DATA}.chunk_04 --output $DATA.chunk_04.zh --src_lang $SRCLG --trg_lang ru --batch_size 32

cat $DATA.chunk_00.de $DATA.chunk_01.en $DATA.chunk_02.fr $DATA.chunk_03.zh $DATA.chunk_04.zh > $DATA.multi

echo "Delete cached chunked file..."
rm $DATA.chunk_*