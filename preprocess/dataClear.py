import os
import sys
import json
import argparse

sys.path.append("..")
from utils.logConfig import Log

def readJson(name):
    data = []
    with open(name, encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    logger.info('Load File %s with %d examples', name, len(data))

    return data

def saveJson(data, name, cont):
    logger.info('Saving to File %s with %d examples', name, len(data))
    mode = 'a' if cont else 'w'
    with open(name, mode, encoding="utf-8") as f:
        cnt = 1
        for d in data:
            d['id'] = cnt
            f.write(json.dumps(d, ensure_ascii=False) + '\n')
            f.flush()
            cnt += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, default="data.txt", help='original file')
    parser.add_argument('-o', type=str, default="cleardaata.txt", help='output file')
    parser.add_argument('-l', type=str, default="en", help='language')
    parser.add_argument('-t', type=float, default=0.5, help='language detection threshold')
    parser.add_argument('-c', action='store_true', default=False, help='language detection threshold')
    args = parser.parse_args()

    logger = Log.getLogger(os.path.basename(sys.argv[0]), "%s.log" % args.l)
    logger.info(args)

    if not os.path.exists(args.i):
        logger.error("%s does not exist", args.i)
        sys.exit()

    oridata = readJson(args.i)
    if 'langdetect' in oridata[0].keys():
        data = [d for d in oridata if d['langdetect'][0][1] > args.t and d['langdetect'][1][1] > args.t and d['langdetect'][2][1] > args.t]
    else:
        data = oridata
    logger.info('After langdetect confidence filter, the remaining example number is %d', len(data))
    data = [d for d in data if len(d["content"])>1 and len(d["summary"])>1]  # empty content and summary
    logger.info('After empty content and summary filter, the remaining example number is %d', len(data))
    data = [d for d in data if d["summary"] != '...']
    logger.info('After meaningless summary filter, the remaining example number is %d', len(data))

    saveJson(data, args.o, args.c)

