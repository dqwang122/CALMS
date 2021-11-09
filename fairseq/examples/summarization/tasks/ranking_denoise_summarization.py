# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
from argparse import Namespace
import json
import itertools
import logging
import os

from fairseq import metrics, options, utils
from fairseq.data import (
    AppendTokenDataset,
    ConcatDataset,
    data_utils,
    encoders,
    indexed_dataset,
    LanguagePairDataset,
    PrependTokenDataset,
    StripTokenDataset,
    TruncateDataset,
    ContrastiveDataset,
    SentenceReplaceDataset
)

from fairseq.tasks.translation_from_pretrained_bart import TranslationFromPretrainedBARTTask
from .summarization_from_pretrained_wo_langtag import load_langpair_sumdataset
from fairseq.tasks import register_task

logger = logging.getLogger(__name__)

@register_task('ranking_denoise_summaization')
class RankDenoiseSummarizationTask(TranslationFromPretrainedBARTTask):
    """
    Translate from source language to target language with a model initialized with a multilingual pretrain.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.

    The translation task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        TranslationFromPretrainedBARTTask.add_args(parser)
        parser.add_argument('--negative-sample-number', type=int, default=1, help='negative sample number for contrastive learning')
        parser.add_argument('--replace-ratio', type=float, default=0.2, help='replace sentence ratio')
        # fmt: on

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args, src_dict, tgt_dict)

        logger.info("bos %d, pad %d, eos %d, unk %d", 
                src_dict.index('<s>'),src_dict.index('<pad>'),
                src_dict.index('</s>'),src_dict.index('<unk>')
                )
        

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = self.args.data.split(':')
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]

        # src="doc", tgt="sum"
        src, tgt = self.args.source_lang, self.args.target_lang

        summ_ds = load_langpair_sumdataset(
            data_path, split, 
            src, self.src_dict, 
            tgt, self.tgt_dict, 
            combine=combine, dataset_impl=self.args.dataset_impl,
            upsample_primary=self.args.upsample_primary,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=getattr(self.args, 'max_source_positions', 1024),
            max_target_positions=getattr(self.args, 'max_target_positions', 1024),
            truncate_source=self.args.truncate_source,
            load_alignments=self.args.load_alignments,
            prepend_bos=getattr(self.args, 'preprend_bos', False),
            append_source_id=True
            )
        
        leadfile = os.path.join(data_path, '{}.lead2'.format(split))
        leads = data_utils.load_indexed_dataset(leadfile, self.src_dict, self.args.dataset_impl)
        rank_ds = ContrastiveDataset(summ_ds, self.src_dict, self.args.seed, self.args.negative_sample_number)
        noisy_ds = SentenceReplaceDataset(rank_ds, leads, self.src_dict, self.args.seed, self.args.replace_ratio)
        

        self.datasets[split] = noisy_ds
        print(self.datasets[split][0])

    def build_model(self, args):
        model = super().build_model(args)

        if getattr(args, 'ranking_head_name', None):
            model.register_classification_head(
                getattr(args, 'ranking_head_name', 'sentence_contrastive_head'),
                num_classes=1,
            )

        return model


    def build_generator(self, models, args):
        if getattr(args, 'score_reference', False):
            from fairseq.sequence_scorer import SequenceScorer
            return SequenceScorer(
                self.target_dictionary,
                # eos=self.tgt_dict.index('[{}]'.format(self.args.sum_lang))
            )
        else:
            from fairseq.sequence_generator import SequenceGenerator
            return SequenceGenerator(
                models,
                self.target_dictionary,
                beam_size=getattr(args, 'beam', 5),
                max_len_a=getattr(args, 'max_len_a', 0),
                max_len_b=getattr(args, 'max_len_b', 200),
                min_len=getattr(args, 'min_len', 1),
                normalize_scores=(not getattr(args, 'unnormalized', False)),
                len_penalty=getattr(args, 'lenpen', 1),
                unk_penalty=getattr(args, 'unkpen', 0),
                temperature=getattr(args, 'temperature', 1.),
                match_source_len=getattr(args, 'match_source_len', False),
                no_repeat_ngram_size=getattr(args, 'no_repeat_ngram_size', 0),
                # eos=self.tgt_dict.index('[{}]'.format(self.args.sum_lang))  # eos: beginning of sentence token
            )

    def build_dataset_for_inference(self, src_tokens, src_lengths):
        src_lang_id = self.source_dictionary.index('[{}]'.format(self.args.doc_lang))
        source_tokens = []
        for s_t in src_tokens:
            s_t = torch.cat([s_t, s_t.new(1).fill_(src_lang_id)])
            source_tokens.append(s_t)
        dataset = LanguagePairDataset(source_tokens, src_lengths, self.source_dictionary)
        return dataset
