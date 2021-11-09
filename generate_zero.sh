#!/usr/bin/env bash

# bash generate_zero.sh /home/dqwang/multilingual/data-bin/de zh loadmodel


########################### Read the configs ###########################

RESOURCE="$1"
LG="$2"
NAME="$3"

argslist=""
for (( i = 4; i <= $# ; i++ ))
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
cp -r $RESOURCE/* $local_dataset_path

output_path=${local_root}/output
model_path=${local_root}/model
mkdir -p ${output_path}
mkdir -p ${model_path}

local_tensorboard_path=${output_path}/tensorboard_logdir  # save the log here
local_checkpoint_path=${output_path}/checkpoint_path      # save the checkpoint here


########################### Start the mode ###########################

langs=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN

echo "Generating..."

suffix=$(echo "$argslist" | sed -e "s/-//g"  -e "s/  */_/g")

python fairseq/generate.py ${local_dataset_path}  \
--path ${local_checkpoint_path}/checkpoint_best.pt \
--task summarization_from_pretrained_wo_langtag \
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
