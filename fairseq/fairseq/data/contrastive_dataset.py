# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import numpy as np
import torch
import math

from . import data_utils, FairseqDataset

logger = logging.getLogger(__name__)


def collate(
    samples, pad_idx, eos_idx, negative_sample_number, left_pad_source=True, left_pad_target=False,
    input_feeding=True,
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
        )

    def check_alignment(alignment, src_len, tgt_len):
        if alignment is None or len(alignment) == 0:
            return False
        if alignment[:, 0].max().item() >= src_len - 1 or alignment[:, 1].max().item() >= tgt_len - 1:
            logger.warning("alignment size mismatch found, skipping alignment!")
            return False
        return True

    def compute_alignment_weights(alignments):
        """
        Given a tensor of shape [:, 2] containing the source-target indices
        corresponding to the alignments, a weight vector containing the
        inverse frequency of each target index is computed.
        For e.g. if alignments = [[5, 7], [2, 3], [1, 3], [4, 2]], then
        a tensor containing [1., 0.5, 0.5, 1] should be returned (since target
        index 3 is repeated twice)
        """
        align_tgt = alignments[:, 1]
        _, align_tgt_i, align_tgt_c = torch.unique(align_tgt, return_inverse=True, return_counts=True)
        align_weights = align_tgt_c[align_tgt_i[np.arange(len(align_tgt))]]
        return 1. / align_weights.float()

    id = torch.LongTensor([s['id'] for s in samples])
    src_tokens = merge('source', left_pad=left_pad_source)
    # sort by descending source length
    src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)

    
    positive_tokens = merge('positive', left_pad=left_pad_source)
    positive_tokens = positive_tokens.index_select(0, sort_order)
    margin_tokens = [positive_tokens.unsqueeze(1)]
    for i in range(negative_sample_number):
        tokens = merge('negative{}'.format(i), left_pad=left_pad_source)
        tokens = tokens.index_select(0, sort_order)
        margin_tokens.append(tokens.unsqueeze(1))
    margin_tokens = torch.cat(margin_tokens, axis=1)
    # print(margin_tokens.size())

    label = merge('label', left_pad=False)
    label = label.index_select(0, sort_order)

    prev_output_tokens = None
    target = None
    if samples[0].get('target', None) is not None:
        target = merge('target', left_pad=left_pad_target)
        target = target.index_select(0, sort_order)
        tgt_lengths = torch.LongTensor([s['target'].numel() for s in samples]).index_select(0, sort_order)
        ntokens = sum(len(s['target']) for s in samples)

        if input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                'target',
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
            )
            prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
    else:
        ntokens = sum(len(s['source']) for s in samples)

    batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
            'margin': margin_tokens,
        },
        'target': target,
        'label': label,
    }

    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens

    if samples[0].get('alignment', None) is not None:
        bsz, tgt_sz = batch['target'].shape
        src_sz = batch['net_input']['src_tokens'].shape[1]

        offsets = torch.zeros((len(sort_order), 2), dtype=torch.long)
        offsets[:, 1] += (torch.arange(len(sort_order), dtype=torch.long) * tgt_sz)
        if left_pad_source:
            offsets[:, 0] += (src_sz - src_lengths)
        if left_pad_target:
            offsets[:, 1] += (tgt_sz - tgt_lengths)

        alignments = [
            alignment + offset
            for align_idx, offset, src_len, tgt_len in zip(sort_order, offsets, src_lengths, tgt_lengths)
            for alignment in [samples[align_idx]['alignment'].view(-1, 2)]
            if check_alignment(alignment, src_len, tgt_len)
        ]

        if len(alignments) > 0:
            alignments = torch.cat(alignments, dim=0)
            align_weights = compute_alignment_weights(alignments)

            batch['alignments'] = alignments
            batch['align_weights'] = align_weights

    return batch


class ContrastiveDataset(FairseqDataset):
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
        vocab,
        seed,
        negative_sample_number,
        eos=None
    ):
        self.dataset = dataset

        self.sizes = dataset.sizes

        self.vocab = vocab
        self.seed = seed
        self.eos = (eos if eos is not None else vocab.eos())
        self.negative_sample_number = negative_sample_number
    
        self.sent_split_tag = self.vocab.eos()

        self.epoch = 0

    def set_epoch(self, epoch, **unused):
        self.epoch = epoch

    def __getitem__(self, index):
        with data_utils.numpy_seed(self.seed, self.epoch, index):
            example = self.dataset[index]
            source, target = example['source'], example['target']

            if self.negative_sample_number > 0:
                candidates = self.get_candidate(source, self.negative_sample_number)
                example.update(self.rank_candidate(candidates, target, number=self.negative_sample_number))
                example['label'] = torch.zeros(1, dtype=torch.long)
                # example['label'] = torch.zeros(self.negative_sample_number+1, dtype=torch.long)
                # example['label'][0] = 1

        return example 

    def __len__(self):
        return len(self.dataset)

    def get_candidate(self, source, number=1):
        full_stops = (source == self.sent_split_tag)
        full_stops[-2] = 1

        sentence_start = (full_stops[1:] * ~full_stops[:-1]).nonzero() + 2

        num_sentences = sentence_start.size(0)
        if num_sentences < number+1:
            logger.info("Sample with num_sentences {}".format(num_sentences))

        candidate_idx = torch.randperm(num_sentences)[:number+1]    # +1 for positive
        candidates = []
        for idx in candidate_idx:
            # ignore language and <bos>
            sent_tokens = source[(sentence_start[idx - 1] if idx > 0 else 2):sentence_start[idx]]
            mask = torch.zeros(len(source), dtype=torch.bool)
            mask[(sentence_start[idx - 1] if idx > 0 else 2):sentence_start[idx]] = True
            candidates.append({'sentence':sent_tokens, 'mask': ~mask})   # 1 means mask
        # print(candidates)
        return candidates


    def rank_candidate(self, candidate, target, number=1):
        
        full_stops = (target == self.sent_split_tag)
        full_stops[-2] = 1
        sentence_start = (full_stops[1:] * ~full_stops[:-1]).nonzero() + 2
        summary_tokens = target[2:sentence_start[-1]]

        

        max_score_idx = 0
        max_score = 0.0
        scores = []
        # ignore language and <bos>
        for i in range(len(candidate)):
            score = cal_overlap(candidate[i]['sentence'], summary_tokens, self.vocab)
            scores.append(score)
            if score > max_score:
                max_score = score
                max_score_idx = i
        # print(scores)

        sample = {'positive': candidate[max_score_idx]['mask']}
        cnt = 0
        for i in range(len(candidate)):
            if i != max_score_idx:
                sample['negative{}'.format(cnt)] = candidate[i]['mask']
                cnt += 1

        if cnt < number:
            for i in range(cnt, number):
                if cnt == 0:
                    sample['negative{}'.format(i)] = candidate[0]['mask']
                else:
                    sample['negative{}'.format(i)] = sample['negative{}'.format(cnt-1)]
        
        return sample


    def collater(self, samples):
        # For now only supports datasets with same underlying collater implementations
        return collate(
            samples, negative_sample_number=self.negative_sample_number, pad_idx=self.dataset.src_dict.pad(), eos_idx=self.eos,
            left_pad_source=self.dataset.left_pad_source, left_pad_target=self.dataset.left_pad_target,
            input_feeding=self.dataset.input_feeding,
        )

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