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
)

from fairseq.tasks.translation_from_pretrained_bart import TranslationFromPretrainedBARTTask
from fairseq.tasks import register_task

logger = logging.getLogger(__name__)

def load_langpair_sumdataset(
    data_path, split,
    src, src_dict,
    tgt, tgt_dict, 
    combine, dataset_impl, upsample_primary,
    left_pad_source, left_pad_target, max_source_positions,
    max_target_positions, prepend_bos=False, load_alignments=False,
    truncate_source=False, append_source_id=False,
):

    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(data_path, '{}.{}-{}.{}'.format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    src_datasets = []
    tgt_datasets = []

    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else '')

        # infer langcode
        if split_exists(split_k, src, tgt, src, data_path):
            prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, src, tgt))
        elif split_exists(split_k, tgt, src, src, data_path):
            prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, tgt, src))
        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError('Dataset not found: {} ({})'.format(split, data_path))

        src_dataset = data_utils.load_indexed_dataset(prefix + src, src_dict, dataset_impl)
        src_datasets.append(src_dataset)

        tgt_dataset = data_utils.load_indexed_dataset(prefix + tgt, tgt_dict, dataset_impl)
        if tgt_dataset is not None:
            tgt_datasets.append(tgt_dataset)

        logger.info('{} {} {}-{} {} examples'.format(
            data_path, split_k, src, tgt, len(src_datasets[-1])
        ))

        if not combine:
            break

    assert len(src_datasets) == len(tgt_datasets) or len(tgt_datasets) == 0

    if len(src_datasets) == 1:
        src_dataset = src_datasets[0]
        tgt_dataset = tgt_datasets[0] if len(tgt_datasets) > 0 else None
    else:
        sample_ratios = [1] * len(src_datasets)
        sample_ratios[0] = upsample_primary
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        if len(tgt_datasets) > 0:
            tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)
        else:
            tgt_dataset = None

    if prepend_bos:
        assert hasattr(src_dict, "bos_index") and hasattr(tgt_dict, "bos_index")
        src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())
        if tgt_dataset is not None:
            tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.bos())

    if truncate_source:
        trunc_len = max_source_positions-1 if append_source_id else max_source_positions
        logger.info("Truncate source to max length %d", trunc_len)
        src_dataset = AppendTokenDataset(
            TruncateDataset(
                StripTokenDataset(src_dataset, src_dict.eos()),
                trunc_len - 1,
            ),
            src_dict.eos(),
        )

    eos = None
    # if append_source_id:
        # src_dataset = AppendTokenDataset(src_dataset, src_dict.index('[{}]'.format(src_lang)))
        # if tgt_dataset is not None:
        #     if langtag_before:
        #         logger.info("Prepend langcode before tgt")
        #         tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.index('[{}]'.format(tgt_lang)))
        #     else:
        #         tgt_dataset = AppendTokenDataset(tgt_dataset, tgt_dict.index('[{}]'.format(tgt_lang)))
        # eos = tgt_dict.index('[{}]'.format(tgt_lang))
    

    align_dataset = None
    if load_alignments:
        align_path = os.path.join(data_path, '{}.align.{}-{}'.format(split, src, tgt))
        if indexed_dataset.dataset_exists(align_path, impl=dataset_impl):
            align_dataset = data_utils.load_indexed_dataset(align_path, None, dataset_impl)

    tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None
    return LanguagePairDataset(
        src_dataset, src_dataset.sizes, src_dict,
        tgt_dataset, tgt_dataset_sizes, tgt_dict,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        max_source_positions=max_source_positions,
        max_target_positions=max_target_positions,
        align_dataset=align_dataset, eos=eos
    )

@register_task('summarization_from_pretrained_wo_langtag')
class SummarizationFromPretrainedMBARTTaskWOLangtag(TranslationFromPretrainedBARTTask):
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
        parser.add_argument('--langtag-before', action='store_true', help='put langtag at the beginning of tgt')
        parser.add_argument('--fixed-encoder-layer', action='store_true', help='fixed encoder layer during from pretrain')
        parser.add_argument('--fixed-decoder-layer', action='store_true', help='fixed decoder layer during from pretrain')
        # fmt: on

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args, src_dict, tgt_dict)
        logger.info("bos %d, pad %d, eos %d, unk %d", 
                src_dict.index('<s>'),src_dict.index('<pad>'),
                src_dict.index('</s>'),src_dict.index('<unk>')
                )
        logger.info("en_XX {}, zh_CN {}, de_DE {}, fr_XX {}".format(
                src_dict.index('[en_XX]'),src_dict.index('[zh_CN]'),
                src_dict.index('[de_DE]'),src_dict.index('[fr_XX]'))
                )
        

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = self.args.data.split(':')
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        src, tgt = self.args.source_lang, self.args.target_lang

        self.datasets[split] = load_langpair_sumdataset(
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
            append_source_id=False
            )
        print(self.datasets[split][0])

    def build_model(self, args):
        model = super().build_model(args)

        if getattr(args, "fixed_encoder_layer", False):
            logger.info("Fix Encoder Layers!")
            for name, param in model.named_parameters():
                if name.startswith("encoder.layers"):
                    param.requires_grad = False

        if getattr(args, "fixed_decoder_layer", False):
            logger.info("Fix Decoder Layers!")
            for name, param in model.named_parameters():
                if name.startswith("decoder.layers"):
                    param.requires_grad = False
        
        return model


    def build_generator(self, models, args):
        if getattr(args, 'score_reference', False):
            from fairseq.sequence_scorer import SequenceScorer
            return SequenceScorer(
                self.target_dictionary,
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
