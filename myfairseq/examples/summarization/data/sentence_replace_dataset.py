# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import numpy as np
import torch
import math

from . import data_utils, FairseqDataset, BaseWrapperDataset

logger = logging.getLogger(__name__)

class SentenceReplaceDataset(FairseqDataset):
    """
    A wrapper around TokenBlockDataset for BART dataset.

    Args:
        dataset (TokenBlockDataset): dataset to wrap
        sizes (List[int]): sentence lengths
        vocab (~fairseq.data.Dictionary): vocabulary
        shuffle (bool, optional): shuffle the elements before batching.
          Default: ``True``
        seed: Seed for random number generator for reproducibility.
        args: argparse arguments.
    """

    def __init__(
        self,
        dataset,
        mapping,
        vocab,
        seed,
        ratio,
        eos=None
    ):
        self.dataset = dataset
        self.src_dict = getattr(dataset, 'src_dict', None)
        self.tgt_dict = getattr(dataset, 'tgt_dict', None)
        self.left_pad_source = getattr(dataset, 'left_pad_source', None)
        self.left_pad_target = getattr(dataset, 'left_pad_target', None)
        self.input_feeding = getattr(dataset, 'input_feeding', None)
        self.mapping = mapping

        self.sizes = dataset.sizes

        self.vocab = vocab
        self.seed = seed
        self.eos = (eos if eos is not None else vocab.eos())
        self.ratio = ratio
    
        self.sent_split_tag = self.vocab.eos()

        self.epoch = 0

    def set_epoch(self, epoch, **unused):
        self.epoch = epoch

    def __getitem__(self, index):
        with data_utils.numpy_seed(self.seed, self.epoch, index):
            example = self.dataset[index]
            lead = self.mapping[index]
            source = example['source']

            if torch.rand(1) <= self.ratio:
                noisy_doc = self.replaceSentence(source, lead)
                example['source'] = noisy_doc

        return example 

    def __len__(self):
        return len(self.dataset)

    def replaceSentence(self, source, lead):
        # source sentence
        full_stops = (source == self.sent_split_tag)
        full_stops[-2] = 1
        doc_sentence_start = (full_stops[1:] * ~full_stops[:-1]).nonzero() + 2

        # lead sentence
        full_stops = (lead == self.sent_split_tag)
        full_stops[-2] = 1
        lead_sentence_start = (full_stops[1:] * ~full_stops[:-1]).nonzero() + 2

        doc_num_sents = doc_sentence_start.size(0)
        lead_num_sents = lead_sentence_start.size(0)

        result = source.clone()

        insert_idx = torch.randperm(doc_num_sents+1)[:lead_num_sents]   
        sents = []
        for i in range(lead_num_sents, doc_num_sents):
            s = source[(doc_sentence_start[i - 1] if i > 0 else 2):doc_sentence_start[i]]
            sents.append(s)
        for i in range(lead_num_sents):
            s = lead[(lead_sentence_start[i - 1] if i > 0 else 1):lead_sentence_start[i]]
            sents.insert(insert_idx[i], s)

        index = 2 # ignore language and <bos>
        for s in sents:
            length = min(s.size(0), result.size(0) - index)
            if length > 0:
                result[index: index+length] = s[:length]
                index += length
            else:
                break
 
        return result



    def collater(self, samples):
        # For now only supports datasets with same underlying collater implementations
        return self.dataset.collater(samples)

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return self.dataset.num_tokens(index)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return self.dataset.size(index)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        return self.dataset.ordered_indices()


def n_grams(tokens, n):
    l = len(tokens)
    return [tuple(tokens[i:i + n]) for i in range(l) if i + n < l]

def cal_overlap(candidate_tokens, summary_tokens,vocab):
    overlap_ratio = []
    for i in range(1, 3):
        summary_ngram = n_grams(summary_tokens, i)
        candidate_ngram = n_grams(candidate_tokens, i)
        overlap = [x for x in candidate_ngram if x in summary_ngram]
        recall = len(overlap) / (len(summary_ngram) + 1e-6)
        overlap_ratio.append(recall)
    return sum(overlap_ratio) / float(len(overlap_ratio))
