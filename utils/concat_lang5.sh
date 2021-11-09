#!/usr/bin/env bash

DIR="/home/tiger/MLSUM/MSPM"
LANG="de en fr zh ru"

get_seeded_random()
{
  seed="$1"
  openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt \
    </dev/zero 2>/dev/null
}

# cat $DIR/de/train.de.spm.doc $DIR/en/train.en.spm.doc $DIR/fr/train.fr.spm.doc $DIR/zh/train.zh.spm.doc $DIR/ru/train.ru.spm.doc > $DIR/lang5/train.noshuffle.spm.doc
# cat $DIR/de/train.de.spm.sum $DIR/en/train.en.spm.sum $DIR/fr/train.fr.spm.sum $DIR/zh/train.zh.spm.sum $DIR/ru/train.ru.spm.sum > $DIR/lang5/train.noshuffle.spm.sum
echo "shuffling"
shuf --random-source=<(get_seeded_random 66) $DIR/lang5/train.noshuffle.spm.doc > $DIR/lang5/train.lang5.spm.doc
shuf --random-source=<(get_seeded_random 66) $DIR/lang5/train.noshuffle.spm.sum > $DIR/lang5/train.lang5.spm.sum

# cat $DIR/de/dev.de.spm.doc $DIR/en/dev.en.spm.doc $DIR/fr/dev.fr.spm.doc $DIR/zh/dev.zh.spm.doc $DIR/ru/dev.ru.spm.doc > $DIR/lang5/dev.noshuffle.spm.doc
# cat $DIR/de/dev.de.spm.sum $DIR/en/dev.en.spm.sum $DIR/fr/dev.fr.spm.sum $DIR/zh/dev.zh.spm.sum $DIR/ru/dev.ru.spm.sum > $DIR/lang5/dev.noshuffle.spm.sum
echo "shuffling"
shuf --random-source=<(get_seeded_random 66) $DIR/lang5/dev.noshuffle.spm.doc > $DIR/lang5/dev.lang5.spm.doc
shuf --random-source=<(get_seeded_random 66) $DIR/lang5/dev.noshuffle.spm.sum > $DIR/lang5/dev.lang5.spm.sum