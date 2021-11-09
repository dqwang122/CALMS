import os
import re
import sys
import json
import argparse

sys.path.append("..")
from utils.logConfig import Log

from string import punctuation as enpunc
from zhon.hanzi import punctuation as zhpunc

def clearPunc(text):
    punc = zhpunc + enpunc
    return re.sub("r[%s]+" % punc, " ", text)

def compContent(train, test):
    testset = {}
    overlaps = []
    line = 1
    for d in test:
        item = " ".join([clearPunc(x) for x in d.split("<q>")])
        testset[item] = line
        line += 1
    line = 1
    for d in train:
        item = " ".join([clearPunc(x) for x in d.split("<q>")])
        if item in testset.keys():
            overlaps.append((line, testset[item]))
        line += 1
    logger.info(overlaps)
    logger.info("There are %d overlaps between the two files", len(overlaps))
    return overlaps

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-tr', type=str, default="zh.doc", help='original file')
    parser.add_argument('-te', type=str, default="zh.sum", help='output dir')
    parser.add_argument('-l', type=str, default="zh", help='language')
    args = parser.parse_args()

    logger = Log.getLogger(os.path.basename(sys.argv[0]), "%s.log" % args.l)
    logger.info(args)

    tr = open(args.tr).readlines()
    te = open(args.te).readlines()
    compContent(tr, te)










