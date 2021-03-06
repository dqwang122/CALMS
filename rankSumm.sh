#!/usr/bin/env bash

# bash rankSumm.sh train lang5 testmodel /home/dqwang/multilingual/data-bin/lang5

########################### Read the configs ###########################

MODE="$1"
LG="$2"
NAME="$3"
RESOURCE="$4"

argslist=""
for (( i = 5; i <= $# ; i++ ))
  do
    j=${!i}
    argslist="${argslist} $j "
  done
echo $argslist >&2


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
cp -r $RESOURCE/* $local_dataset_path

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
    --task ranking_summaization \
    --arch rank_summ_large \
    --source-lang doc --target-lang sum \
    --langs $langs \
    --dataset-impl mmap \
    --truncate-source \
    --encoder-normalize-before --decoder-normalize-before \
    --layernorm-embedding \
    --criterion cross_entropy_and_ranking --label-smoothing 0.2 \
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
    --patience 3 \
    --user-dir examples/summarization \
    --ranking-head-name sentence_contrastive_head \
    --negative-sample-number 1 \
    --ranking-loss-weight 1 \
    $argslist

fi
