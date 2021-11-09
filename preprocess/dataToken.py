import os
import sys
import argparse

sys.path.append("..")
from utils.logConfig import Log
from utils.tokenizer import MosesTokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, default="test.txt", help='input file')
    parser.add_argument('-o', type=str, default="test.tokenize.txt", help='output file')
    parser.add_argument('-l', type=str, default="en", help='language')
    parser.add_argument('-d', type=str, default="<q>", help='delimiter')
    args = parser.parse_args()

    logger = Log.getLogger(os.path.basename(sys.argv[0]), "%s.log" % args.l)
    logger.info(args)

    mose = MosesTokenizer(args.l)

    with open(args.i, 'r') as fin, open(args.o, 'w') as fout:
        for line in fin:
            sents = line.strip().split('<q>')
            sents = [mose.tokenize(s) for s in sents]
            doc = "<q>".join(sents)
            fout.write(doc + "\n")
            fout.flush()