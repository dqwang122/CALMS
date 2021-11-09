import re
import os
import sys
import json
import numpy
import random
import argparse

from utils.tokenizer import SBDSplitor

sys.path.append("..")
from utils.logConfig import Log

from string import punctuation as enpunc
from zhon.hanzi import punctuation as zhpunc

MIN_DOC_LEN = 50
MIN_SUM_LEN = 5
MAX_DOC_LEN = 5000
random.seed(233)

def clearPunc(text):
    punc = zhpunc + enpunc
    return re.sub("r[%s]+" % punc, " ", text)

def readJson(name):
    data = []
    with open(name, encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    logger.info('Load File %s with %d examples', name, len(data))
    return data

def saveJson(data, name):
    with open(name, 'w', encoding="utf-8") as fout:
        for k, v in data.items():
            v['id'] = k
            fout.write(json.dumps(v, ensure_ascii=False) + '\n')
            fout.flush()
    logger.info('Saving File %s with %d examples', name, len(data))

def saveTxt(data, key, filename):
    with open(filename, 'w') as fout:
        for d in data:
            if key:
                fout.write(d[key] + '\n')
            else:
                fout.write(d + '\n')
            fout.flush()
    logger.info('Saving %s into File %s', key, filename)

def convertPunc(x):
    x = x.replace('「', '“').replace('」', '”')
    x = x.replace('『', '‘').replace('』', '’')
    return x

def clear(content):
    content = [x.strip().replace("\r", "").replace('\n', ' ') for x in content]
    content = [convertPunc(x) for x in content]
    content = [x for x in content if len(x) > 2]
    return "<q>".join(content)

def resplit(content, splitor):
    sents = content.split('<q>')
    sents = [splitor.split(s) for s in sents]
    sents = sum(sents, [])
    return "<q>".join(sents)

# def uniq(examples):
#     S = {}
#     for id, content in examples.items():
#         item = tuple(content.items())
#         if item not in S.keys():
#             S[item] = id
#     uniqEx = {}
#     for k, v in S.items():
#         uniqEx[v] = dict(k)
#     return uniqEx

def uniq(examples):
    doc = set()
    summ = set()
    ids = []
    for idx, content in examples.items():
        d = clearPunc(content['article'])
        s = clearPunc(content['summary'])
        if d not in doc and s not in summ:
            ids.append(idx)
        doc.add(d)
        summ.add(s)
    uniqEx = {}
    for idx in ids:
        uniqEx[idx] = examples[idx]
    return uniqEx

def getlength(examples, language):
    if language == 'zh' or language == 'ja': # without spaces
        article_len = [len(list(d['article'].replace('<q>', ''))) for d in examples.values()]
        summary_len = [len(list(d['summary'])) for d in examples.values()]
    else:
        article_len = [len(d['article'].replace('<q>', ' ').split()) for d in examples.values()]
        summary_len = [len(d['summary'].split()) for d in examples.values()]
    return article_len, summary_len

def process(data, language):
    examples = dict([(d['id'], {'article': clear(d['content']), 'summary': convertPunc(d['summary']).replace('\n', ' ')}) for d in data])
    examples = uniq(examples)
    logger.info('Uniq example number is %d', len(examples))
    article_len, summary_len = getlength(examples, language)
    examples = dict([(idx, examples[idx]) for idx, doc, summ in zip(examples.keys(), article_len, summary_len) if doc>=MIN_DOC_LEN and summ>=MIN_SUM_LEN and doc>=summ])
    logger.info('Example number with more than %d doc length and %d sum length is %d', MIN_DOC_LEN, MIN_SUM_LEN, len(examples))
    article_len, summary_len = getlength(examples, language)
    examples = dict([(idx, examples[idx]) for idx, doc in zip(examples.keys(), article_len) if doc <= MAX_DOC_LEN])
    logger.info('Example number with shorter than %d doc length is %d', MAX_DOC_LEN, len(examples))
    article_len, summary_len = getlength(examples, language)
    logger.info('Article length\t%d\t%d\t%d', min(article_len), max(article_len), sum(article_len)/len(article_len))
    logger.info('Summary length\t%d\t%d\t%d', min(summary_len), max(summary_len), sum(summary_len)/len(summary_len))
    sbd = SBDSplitor(language)
    examples = dict([(idx, {'article': resplit(value['article'], sbd), 'summary': resplit(value['summary'], sbd)}) for idx, value in examples.items()])
    summary_sents = [len(d['summary'].split('<q>')) for d in examples.values()]
    logger.info('Summary sentences\t%d\t%d\t%.2f', min(summary_sents), max(summary_sents), sum(summary_sents) / len(summary_sents))
    return examples

def splitData(data):
    idlist = {}
    ids = list(data.keys())
    random.shuffle(ids)
    idlist['train'] = ids[:int(len(data) * 0.9)]
    idlist['dev'] = ids[int(len(data) * 0.9): int(len(data) * 0.95)]
    idlist['test'] = ids[int(len(data) * 0.95):]
    logger.info('Train/Dev/Test\t%d\t%d\t%d', len(idlist['train']), len(idlist['dev']), len(idlist['test']))
    return idlist

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, default="data.txt", help='original file')
    parser.add_argument('-d', type=str, default="source", help='output dir')
    parser.add_argument('-l', type=str, default="en", help='language')
    parser.add_argument('-p', action='store_true', help='-p indicates to split the dataset')
    args = parser.parse_args()

    if not os.path.exists(args.d):
        os.mkdir(args.d)

    logger = Log.getLogger(os.path.basename(sys.argv[0]), "%s.log" % args.l)
    logger.info(args)

    oridata = readJson(args.i)
    logger.info('The total number of %s is %d', args.i, len(oridata))
    examples = process(oridata, args.l)
    saveJson(examples, os.path.join(args.d, "%s.jsonl" % args.l))

    sentline = [e['article'].replace('<q>', '\n') for e in examples.values()]
    sentline.extend([e['summary'] for e in examples.values()])
    saveTxt(sentline, None, os.path.join(args.d, "lines.txt"))

    idlist = splitData(examples) if args.p else {'test': list(examples.keys())}
    json.dump(idlist, open(os.path.join(args.d, 'idlist.%s' % args.l), 'w'), indent=2, separators=(',', ':'), ensure_ascii=False)
    logger.info('Saving idlist into File %s', os.path.join(args.d, 'idlist.%s' % args.l))

    for k in idlist.keys():
        dataset = [examples[idx] for idx in idlist[k]]
        saveTxt(dataset, 'article', os.path.join(args.d, "%s.%s.doc" % (k, args.l)))
        saveTxt(dataset, 'summary', os.path.join(args.d, "%s.%s.sum" % (k, args.l)))
