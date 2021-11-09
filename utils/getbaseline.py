import os
import time
import random
import datetime
import re
import argparse
import sys

import json

sys.path.append("..")
from utils.calRouge import pyrouge_score
from utils.logConfig import Log
from utils.tokenizer import MosesTokenizer, SBDSplitor

random.seed(144)


def loaddata(path, idpath, key):
    data = []
    idlist = json.load(open(idpath))
    ids = idlist[key]
    with open(path) as f:
        for line in f:
            e = json.loads(line)
            if e['id'] in ids:
                data.append(e)
    return data


def rougelabel(document, summary, label, language='en'):
    start = time.time()
    hyps_list = []
    refer_list = []
    for doc, summ, l in zip(document, summary, label):
        article_sents = doc
        abstract_sentences = [summ] if isinstance(summ, str) else summ
        refer_list.append("\n".join(abstract_sentences))
        selected_sent = []
        for idx in l:
            selected_sent.append(article_sents[idx])
        hyps_list.append("\n".join(selected_sent))
    logger.info("Start pyrouge for label!")
    scores = pyrouge_score(hyps_list, refer_list, language=language)
    logger.info("The total set is %d, the time cost is %f", len(hyps_list), time.time() - start)
    logger.info("Rouge-1\t%.2f\t%.2f\t%.2f" % (scores['rouge-1']['p'], scores['rouge-1']['r'], scores['rouge-1']['f']))
    logger.info("Rouge-2\t%.2f\t%.2f\t%.2f" % (scores['rouge-2']['p'], scores['rouge-2']['r'], scores['rouge-2']['f']))
    logger.info("Rouge-l\t%.2f\t%.2f\t%.2f" % (scores['rouge-l']['p'], scores['rouge-l']['r'], scores['rouge-l']['f']))
    
def rougelead(document, summary, count, language='en'):
    start = time.time()
    hyps_list = []
    refer_list = []
    for doc, summ in zip(document, summary):
        article_sents = doc
        abstract_sentences = [summ] if isinstance(summ, str) else summ
        refer_list.append("\n".join(abstract_sentences))
        hyps_list.append("\n".join(article_sents[:count]))
    logger.info("Start pyrouge for lead!")
    scores = pyrouge_score(hyps_list, refer_list, language=language)
    logger.info("The total set is %d, the time cost is %f", len(hyps_list), time.time() - start)
    logger.info("Rouge-1\t%.2f\t%.2f\t%.2f" % (scores['rouge-1']['p'], scores['rouge-1']['r'], scores['rouge-1']['f']))
    logger.info("Rouge-2\t%.2f\t%.2f\t%.2f" % (scores['rouge-2']['p'], scores['rouge-2']['r'], scores['rouge-2']['f']))
    logger.info("Rouge-l\t%.2f\t%.2f\t%.2f" % (scores['rouge-l']['p'], scores['rouge-l']['r'], scores['rouge-l']['f']))

    logger.info("Start rouge for lead!")
    try:
        from rouge import Rouge
        rouge = Rouge()
        scores = rouge.get_scores(hyps_list, refer_list, avg=True)
        logger.info("The total set is %d, the time cost is %f", len(hyps_list), time.time() - start)
        logger.info("Rouge-1\t%.2f\t%.2f\t%.2f" % (scores['rouge-1']['p']*100, scores['rouge-1']['r']*100, scores['rouge-1']['f']*100))
        logger.info("Rouge-2\t%.2f\t%.2f\t%.2f" % (scores['rouge-2']['p']*100, scores['rouge-2']['r']*100, scores['rouge-2']['f']*100))
        logger.info("Rouge-l\t%.2f\t%.2f\t%.2f" % (scores['rouge-l']['p']*100, scores['rouge-l']['r']*100, scores['rouge-l']['f']*100))
    except ImportError:
        raise ImportError('Please install fast rouge with: pip install rouge')

    
def rougerandom(document, summary, count, language='en'):
    start = time.time()
    hyps_list = []
    refer_list = []
    for doc, summ in zip(document, summary):
        article_sents = doc
        abstract_sentences = [summ] if isinstance(summ, str) else summ
        refer_list.append("\n".join(abstract_sentences))
        selected_sent = []
        idx_list = random.sample(range(len(article_sents)), min(count, len(article_sents)))
        for idx in idx_list:
            selected_sent.append(article_sents[idx])
        hyps_list.append("\n".join(selected_sent))
    logger.info("Start pyrouge for random!")
    scores = pyrouge_score(hyps_list, refer_list, language=language)
    logger.info("The total set is %d, the time cost is %f", len(hyps_list), time.time() - start)
    logger.info("Rouge-1\t%.2f\t%.2f\t%.2f" % (scores['rouge-1']['p'], scores['rouge-1']['r'], scores['rouge-1']['f']))
    logger.info("Rouge-2\t%.2f\t%.2f\t%.2f" % (scores['rouge-2']['p'], scores['rouge-2']['r'], scores['rouge-2']['f']))
    logger.info("Rouge-l\t%.2f\t%.2f\t%.2f" % (scores['rouge-l']['p'], scores['rouge-l']['r'], scores['rouge-l']['f']))


    """ Usage example:
            python getbaseline.py -i ../data/clean/zh/zh.jsonl -k test -m lead -c 2 -l zh -d "<q>" -idlist ../data/clean/zh/idlist.zh -t
            python -m utils.getbaseline -i data/clean/zh/zh.jsonl -k test -m lead -c 2 -l zh -d "<q>" -idlist data/clean/zh/idlist.zh -t
    """


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, default='test.label.jsonl', help='dataset path')
    parser.add_argument('-k', type=str, default='test', help='dataset split')
    parser.add_argument('-m', '--mode', type=str, default="label", help="lead/random/label")
    parser.add_argument('-c', '--count', type=int, default=3, help='select sentence number')
    parser.add_argument('-l', '--language', type=str, default='zh', help='language')
    parser.add_argument('-d', '--delimiter', type=str, default='<q>', help='delimiter for document')
    parser.add_argument('-t', '--tokenize', action='store_true', help='need to tokenize the original document')
    parser.add_argument('-idlist', type=str, default='idlist.zh', help='idlist path')
    parser.add_argument('-doc', type=str, default='article', help='document key name')
    parser.add_argument('-sum', type=str, default='summary', help='summary key name')
    args = parser.parse_args()

    logger = Log.getLogger(os.path.basename(sys.argv[0]), "%s.log" % args.language)
    logger.info(args)

    data = loaddata(args.i, args.idlist, args.k)

    document = []
    summary = []
    label = []

    mose = MosesTokenizer(args.language)
    sbd = SBDSplitor(args.language)

    for e in data:
        if args.delimiter:
            doc = e[args.doc].split(args.delimiter) if isinstance(e[args.doc], str) else e[args.doc]
            summ = e[args.sum].split(args.delimiter) if isinstance(e[args.sum], str) else e[args.sum]
        else:
            doc = sbd.split(e[args.doc]) if isinstance(e[args.doc], str) else e[args.doc]
            summ = sbd.split(e[args.sum]) if isinstance(e[args.sum], str) else e[args.sum]
        if args.tokenize:
            doc = [mose.tokenize(d) for d in doc]
            summ = [mose.tokenize(d) for d in summ]
        document.append(doc)
        summary.append(summ)
        label.append(e.get('label', []))
    summary_len = [len(s) for s in summary]
    logger.info("The average sentence number of summary is %f", sum(summary_len)/len(summary_len))
    logger.info("The data path is %s with %d examples", args.i, len(document))

    if args.mode == 'label':
        eval('rouge{mode}(document, summary, label, language=\'{language}\')'.format(mode=args.mode, language=args.language))
    else:
        eval('rouge{mode}(document, summary, count={count}, language=\'{language}\')'.format(mode=args.mode, count=args.count, language=args.language))
        
        
        
