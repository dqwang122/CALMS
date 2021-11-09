import os
import time
import datetime
import re
import argparse
import sys
import random

import json

sys.path.append("..")
from utils.logConfig import Log

random.seed(666)


def readTxt(name):
    data = []
    with open(name, 'rb') as fin:
        for line in fin:
            data.append(line.decode('utf-8').replace(u'\u2028', '').replace('\r', '').strip())
    logger.info("Reading file %s with %d", name, len(data))
    return data

def saveTxt(name, data):
    logger.info("Saving data with size %d to file %s", len(data), name)
    with open(name, 'w') as fout:
        for d in data:
            fout.write(d + "\n")

def getlead(args, data):
    docs = [doc.split(args.delimiter) for doc in data]
    docs = [doc for doc in docs if len(doc) > args.min]
    leads = [" <q> ".join(doc[:args.count]) for doc in docs]
    docs = [" <q> ".join(doc) for doc in docs ]
    return docs, leads

def getoracle(args, data):
    
    return docs, oracles

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, default='doc.txt', help='dataset path')
    parser.add_argument('-c', '--count', type=int, default=2, help='select sentence number')
    parser.add_argument('-min', type=int, default=10, help='the min len of document')
    parser.add_argument('-m', type=str, default='lead', help='lead/oracle')
    parser.add_argument('-l', '--language', type=str, default='zh', help='language')
    parser.add_argument('-d', '--delimiter', type=str, default='<q>', help='delimiter for document')
    args = parser.parse_args()

    logger = Log.getLogger(os.path.basename(sys.argv[0]), None)
    logger.info(args)

    data = readTxt(args.i)
    docs, results = eval('get{}(data, args)'.format(args.m))
    
    saveTxt(args.i.replace("doc", "doc{}".format(args.min)), docs)
    saveTxt(args.i.replace("doc", "lead{}".format(args.count)), results)
    



    

    