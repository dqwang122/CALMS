from __future__ import division

import re
import copy
import json
import time
import argparse
import numpy as np
from collections import Counter
from nltk.util import ngrams

################# tools #################

def splitChars(sent, lang):
    if lang == 'zh':
        parts = re.split(u"([\u4e00-\u9fa5])", sent)
    elif lang == 'ja':
        parts = re.split(u"([\u0800-\u4e00])",sent)
    elif lang == 'ko':
        parts = re.split(u"([\uac00-\ud7ff])", sent)
    else:   # Chinese, Japanese and Korean non-symbol characters
        parts = re.split(u"([\u2e80-\u9fff])", sent)
    return [p.strip().lower() for p in parts if p != "" and p != " "]

def str2char(string, language='all'):
    sents = string.split("\n")
    tokens = [" ".join(splitChars(s, language)) for s in sents]
    return "\n".join(tokens)

def _modified_precision(candidate, references, n):
    counts = Counter(ngrams(candidate, n))

    if not counts:
        return 0

    max_counts = {}
    for reference in references:
        reference_counts = Counter(ngrams(reference, n))
        for ngram in counts:
            max_counts[ngram] = max(max_counts.get(ngram, 0), reference_counts[ngram])

    clipped_counts = dict((ngram, min(count, max_counts[ngram])) for ngram, count in counts.items())

    return sum(clipped_counts.values()) / sum(counts.values())

def compute_rouge_all(candidate, references, ngram):
    #candidate = [candidate]
    recall = (
        _modified_precision(references, [candidate], i)
        for i in ngram
    )

    precision = (
        _modified_precision(candidate, [references], i)
        for i in ngram
    )

    R, P, F = [], [], []
    for rel, pre in zip(recall, precision):
        R.append(rel)
        P.append(pre)
        F.append(2 * rel * pre / (rel + pre) if (rel+pre != 0) else 0)
    return P, R, F


def rouge_eval(hyps, refer, language='zh'):
    """ modified Rouge with uniq n-gram (faster)

    :param hyps: string
    :param refer: string
    :return: R-1 F1
    """
    hyps = str2char(hyps, language=language).split()
    refer = str2char(refer, language=language).split()
    P, R, F = compute_rouge_all(hyps, refer, [1])
    return F[0]

def rouge_1_recall(hyps, refer, language='zh'):
    """ modified Rouge with uniq n-gram (faster)

    :param hyps: string
    :param refer: string
    :return: R-1 F1
    """
    hyps = str2char(hyps, language=language).split()
    refer = str2char(refer, language=language).split()
    P, R, F = compute_rouge_all(hyps, refer, [1])
    return R[0]

def rouge_2_recall(hyps, refer, language='zh'):
    """ modified Rouge with uniq n-gram (faster)

    :param hyps: string
    :param refer: string
    :return: R-1 F1
    """
    hyps = str2char(hyps, language=language).split()
    refer = str2char(refer, language=language).split()
    P, R, F = compute_rouge_all(hyps, refer, [2])
    return R[0]


# def rouge_eval(hyps, refer):
#     """ rouge from ROUGE package """
#     from rouge import Rouge
#     hyps = str2char(hyps, language='zh')
#     refer = str2char(refer, language='zh')
#     rouge = Rouge()
#     try:
#         score = rouge.get_scores(hyps, refer)[0]
#         mean_score = np.mean([score["rouge-1"]["f"], score["rouge-2"]["f"], score["rouge-l"]["f"]])
#     except:
#         mean_score = 0.0
#     return mean_score



################# Function #################

def calLabel(article, abstract):
    hyps_list = article
    refer = abstract
    scores = []
    for hyps in hyps_list:
        mean_score = rouge_eval(hyps, refer)
        scores.append(mean_score)

    selected = [int(np.argmax(scores))]
    selected_sent_cnt = 1

    best_rouge = np.max(scores)
    while selected_sent_cnt < len(hyps_list):
        cur_max_rouge = 0.0
        cur_max_idx = -1
        for i in range(len(hyps_list)):
            if i not in selected:
                temp = copy.deepcopy(selected)
                temp.append(i)
                hyps = "\n".join([hyps_list[idx] for idx in np.sort(temp)])
                cur_rouge = rouge_eval(hyps, refer)
                if cur_rouge > cur_max_rouge:
                    cur_max_rouge = cur_rouge
                    cur_max_idx = i
        if cur_max_rouge != 0.0 and cur_max_idx != -1 and cur_max_rouge >= best_rouge and len(hyps_list[cur_max_idx]) > 1:
            selected.append(cur_max_idx)
            selected_sent_cnt += 1
            best_rouge = cur_max_rouge
        else:
            break
    print(selected, best_rouge)
    return selected

def calBest(article, abstract, k):
    hyps_list = article
    refer = abstract
    scores = []
    for hyps in hyps_list:
        mean_score = rouge_eval(hyps, refer)
        scores.append(mean_score)

    scores = np.array(scores)
    selected = scores.argsort()[-k:][::-1]
    print(selected, scores[selected])
    return selected.tolist()

def maskDoc(article, abstract, mode="label", k=3):
    """

    :param article: list of string, each element is a sentece
    :param abstract: string
    :param mode: string, label/best
    :param k: int, top-k for mode best
    :return:
        selected idx: list of int
        selected sentences: list of string
    """
    selected = []
    if mode == "label":
        selected = calLabel(article, abstract)
    elif mode == "best":
        selected = calBest(article, abstract, k)
    else:
        print("Mode must be label/best!")

    selected_sent = [article[idx] for idx in selected]
    return selected, selected_sent






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, default='.', help='dataset file')
    parser.add_argument('-k', type=int, default=2, help='top-k for rouge best sentences')
    parser.add_argument('-m', type=str, default='label', help='mask mode [label/best]')
    args = parser.parse_args()

    start = time.time()

    with open(args.i) as f:
        for line in f:
            data = json.loads(line)
            article = data['content']
            abstract = data['title']
            print(maskDoc(article, abstract, args.m, args.k), abstract)

    print("Time: %.5f" % (time.time() - start))



