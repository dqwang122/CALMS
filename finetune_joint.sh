#!/usr/bin/env bash

# bash finetune_joint.sh train lang5 testmodel /home/dqwang/multilingual/data-bin/
# bash finetune_joint.sh generate de testmodel /home/dqwang/multilingual/data-bin/de --doc-lang de_DE --sum-lang de_DE

########################### Read the configs ###########################

MODE="$1"     
LG="$2"   # lang5 for train           
NAME="$3"
RESOURCE="$4"

PRELANG=(de_DE en_XX fr_XX ru_RU zh_CN)

argslist=""
for (( i = 5; i <= $# ; i++ ))
  do
    j=${!i}
    argslist="${argslist} $j "
  done
echo $argslist >&2

cd "$(dirnameget $0)" || return

########################### Build the environment ###########################

cd "$(dirname $0)" || return

echo "Install Requirements" >&2
pip3 install -e fairseq
pip3 install -r requirements.txt

sudo apt-get update
sudo apt-get install libxml-perl libxml-dom-perl

echo "Install ROUGE" >&2
export PYROUGE_HOME_DIR=$(pwd)/RELEASE-1.5.5
export PYROUGE_TEMP_PATH=.
pyrouge_set_rouge_path $PYROUGE_HOME_DIR
chmod +x $PYROUGE_HOME_DIR/ROUGE-1.5.5.pl


########################### Create the workshop ###########################

local_root=~/${NAME}_${LG}
resource_root=${local_root}/resource              # put the resource here
local_dataset_path=${resource_root}/dataset       # put the dataset here
local_pretrained_path=${resource_root}/mbart.cc25 # put the pretrained model here
for LG in ${PRELANG[*]}
  do
    LGCODE=${LG: 0 : 2}
    cp -r $RESOURCE/$LGCODE ${local_dataset_path}
  done
cp ${local_dataset_path}/$LGCODE/dict.*.txt ${local_dataset_path}


output_path=${local_root}/output
model_path=${local_root}/model
mkdir -p ${output_path}
mkdir -p ${model_path}

local_tensorboard_path=${output_path}/tensorboard_logdir  # save the log here
mkdir -p ${local_tensorboard_path}

local_checkpoint_path=${output_path}/checkpoint_path      # save the checkpoint here
mkdir -p ${local_checkpoint_path}


########################### Start the mode ###########################

langs=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN

if [ "$MODE" == "train" ]; then
  echo "Training..."

  python fairseq/train.py ${local_dataset_path} --ddp-backend=no_c10d \
    --save-dir ${local_checkpoint_path} \
    --tensorboard-logdir ${local_tensorboard_path} \
    --restore-file ${local_pretrained_path}/model.pt \
    --task summarization_from_pretrained_mbart_joint \
    --arch mbart_large \
    --source-lang doc --target-lang sum \
    --langs $langs \
    --dataset-impl mmap \
    --truncate-source \
    --encoder-normalize-before --decoder-normalize-before \
    --layernorm-embedding \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.2 \
    --reset-optimizer --reset-dataloader --reset-meters --reset-lr-scheduler \
    --required-batch-size-multiple 1 \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
    --lr 3e-5 --min-lr -1 \
    --lr-scheduler polynomial_decay \
    --clip-norm 0.1 \
    --update-freq 4 \
    --skip-invalid-size-inputs-valid-test \
    --find-unused-parameters \
    --num-workers 100 \
    --fp16 \
    --max-tokens 4096 \
    --total-num-update 200000 --warmup-updates 2500 \
    --log-interval 200 \
    --log-format simple \
    --keep-best-checkpoints 3 \
    --no-epoch-checkpoints \
    --patience 5 \
    --user-dir examples/summarization \
    $argslist


elif [ "$MODE" == "generate" ]; then
  echo "Generating..."

  suffix=$(echo "$argslist" | sed -e "s/-//g"  -e "s/  */_/g")
  
  python fairseq/generate.py ${local_dataset_path}/${LG}  \
  --path ${local_checkpoint_path}/checkpoint_best.pt \
  --task summarization_from_pretrained_mbart_joint \
  --gen-subset test \
  --source-lang doc --target-lang sum \
  --langs $langs \
  --remove-bpe 'sentencepiece'  \
  --min-len 30 \
  --max-len-b 50 \
  --lenpen 0.6 \
  --no-repeat-ngram-size 3 \
  --truncate-source \
  --user-dir examples/summarization \
  $argslist \
  > ${local_tensorboard_path}/"output$suffix"

  cat ${local_tensorboard_path}/"output$suffix" | grep -P "^H" | sort -V |cut -f 3- | sed -e  "s/\[[a-z]\{2\}_[A-Z]\{2\}\]//g" > ${local_tensorboard_path}/"test$suffix.hypo"

  python utils/calRouge.py \
  -c ${local_tensorboard_path}/"test$suffix.hypo" \
  -r ${local_dataset_path}/test.${LG}.sum \
  -l ${LG} -d "<q>"

fi
