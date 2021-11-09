#!/usr/bin/env bash
set -e

# bash predata.sh data zh zh_CN MSPM

DATADIR="$1"
LG="$2"
TAG="$3"
MODE="$4"
DIVISIONS="$5"

if [ -z "$DATADIR" ]; then
    echo "Lose the raw data dir!"
    exit
fi
if [ -z "$LG" ]; then
    echo "Lose the language!"
    exit
fi
if [[ -n "$DIVISIONS" ]] && [[ "$DIVISIONS" == "test" ]]; then
    PARTS=(test)
else
    PARTS=(train dev test)
fi
echo "Parts are : ${PARTS[*]}"

VOCAB_SIZE=32000
DATA="$DATADIR/clean0226/$LG"
TOKEN="$DATADIR/$MODE/$LG"
if [ ! -d "$TOKEN"  ] ; then
    mkdir -p "$TOKEN"
fi

if [[ "$MODE" == "SPM" ]]; then
  if [ ! -f "$DATA/sentence.bpe.$VOCAB_SIZE.$LG.model" ] ; then
    echo "SPM training for Language..."
    if [[ "$LG" == "zh" ]] || [[ "$LG" == "ja" ]] ; then
      python fairseq/scripts/spm_train.py \
      --input="$DATA/lines.txt" \
      --model_prefix="$DATA/sentence.bpe.$VOCAB_SIZE.$LG" \
      --vocab_size=$VOCAB_SIZE \
      --character_coverage=0.995 \
      --model_type=bpe \
      --user_defined_symbols='<q>'
    else
      python fairseq/scripts/spm_train.py \
      --input="$DATA/lines.txt" \
      --model_prefix="$DATA/sentence.bpe.$VOCAB_SIZE.$LG" \
      --vocab_size=$VOCAB_SIZE \
      --character_coverage=1 \
      --model_type=bpe \
      --user_defined_symbols='<q>'
    fi
  fi

  echo "SPM encoding for dataset..."
  for SPLIT in ${PARTS[*]}
    do
      echo "  encoding $DATA/$SPLIT.$LG.doc to $TOKEN/$SPLIT.$LG.spm.doc ..."
      python fairseq/scripts/spm_encode.py --model="$DATA/sentence.bpe.$VOCAB_SIZE.$LG.model" < "$DATA/$SPLIT.$LG.doc" | sed "s/^/<s> /" -e "s/$/ <\/s>/" -e "s/^/[${TAG}] /" > "$TOKEN/$SPLIT.$LG.spm.doc"
      echo "  encoding $DATA/$SPLIT.$LG.sum to $TOKEN/$SPLIT.$LG.spm.sum ..."
      python fairseq/scripts/spm_encode.py --model="$DATA/sentence.bpe.$VOCAB_SIZE.$LG.model" < "$DATA/$SPLIT.$LG.sum" | sed "s/^/<s> /" -e "s/$/ <\/s>/" -e "s/^/[${TAG}] /"> "$TOKEN/$SPLIT.$LG.spm.sum"
    done

  echo "Changing vocab to dict..."
  python -m preprocess.vocab2dict -i "$DATA/sentence.bpe.$VOCAB_SIZE.$LG.vocab" -o "$DATA/$LG.dict.txt" -l "$LG"
  DICT="$DATA/$LG.dict.txt"


elif [[ "$MODE" == "MSPM" ]]; then
  MBART=/mnt/bd/wdq-workshop/pretrain/mbart.cc25
  MODEL=$MBART/sentence.bpe.model
  # DICT=$MBART/dict.txt
  DICT=$MBART/dict_extend.txt


  echo "MSPM encoding for dataset..."
  for SPLIT in ${PARTS[*]}
    do
      echo "  encoding $DATA/$SPLIT.$LG.doc to $TOKEN/$SPLIT.$LG.spm.doc ..."
      python fairseq/scripts/spm_encode.py --model=$MODEL < "$DATA/$SPLIT.$LG.doc" | sed -e "s/< q >/ <\/s>/g" -e "s/^/<s> /" -e "s/$/ <\/s>/" > "$TOKEN/$SPLIT.$LG.spm.doc"
      echo "  encoding $DATA/$SPLIT.$LG.sum to $TOKEN/$SPLIT.$LG.spm.sum ..."
      python fairseq/scripts/spm_encode.py --model=$MODEL < "$DATA/$SPLIT.$LG.sum" | sed -e "s/< q >/ <\/s>/g" -e "s/^/<s> /" -e "s/$/ <\/s>/" > "$TOKEN/$SPLIT.$LG.spm.sum"
    done



elif [[ "$MODE" == "BPE" ]]; then
  wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'
  wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'
  wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt'

  DICT="dict.txt"
  for SPLIT in ${PARTS[*]}
  do
    for TYPE in doc sum
    do
      python -m examples.roberta.multiprocessing_bpe_encoder \
        --encoder-json encoder.json \
        --vocab-bpe vocab.bpe \
        --inputs "$DATA/$SPLIT.$LG.$TYPE" \
        --outputs "$TOKEN/$SPLIT.$LG.bpe.$TYPE" \
        --workers 60 \
        --keep-empty
    done
  done

fi

echo "Generating data-bin for dataset..."
  INPUT="$TOKEN"
  BINDIR="$DATADIR/data-bin/$MODE/$LG"
  if [[ "$MODE" == "SPM" ]] || [[ "$MODE" == "MSPM" ]]; then
    TOKENTYPE="spm"
  else
    TOKENTYPE="bpe"
  fi

  echo "Binarized $INPUT ($TOKENTYPE) to $BINDIR with dict $DICT"

  if [ ${#PARTS[*]} == 3 ]; then
    python fairseq/preprocess.py \
    --source-lang doc \
    --target-lang sum \
    --trainpref "$INPUT/train.$LG.$TOKENTYPE" \
    --validpref "$INPUT/dev.$LG.$TOKENTYPE" \
    --testpref "$INPUT/test.$LG.$TOKENTYPE"  \
    --destdir "$BINDIR" \
    --srcdict "$DICT" \
    --tgtdict "$DICT" \
    --workers 70
  else
    python fairseq/preprocess.py \
      --source-lang doc \
      --target-lang sum \
      --testpref "$INPUT/test.$LG.$TOKENTYPE"  \
      --destdir "$BINDIR" \
      --thresholdtgt 0 \
      --thresholdsrc 0 \
      --srcdict "$DICT" \
      --tgtdict "$DICT" \
      --workers 70
  fi