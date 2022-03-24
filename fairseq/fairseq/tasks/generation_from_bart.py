# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import logging

from fairseq.data import LanguagePairDataset

from fairseq.data import data_utils, FairseqDataset, iterators, Dictionary
from .translation import load_langpair_dataset, TranslationTask
from . import register_task

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@register_task('generation_from_bart')
class GenerationFromBARTTask(TranslationTask):
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
        TranslationTask.add_args(parser)
        parser.add_argument('--langs', required=True, metavar='LANG',
                            help='comma-separated list of monolingual language, for example, "en,de,fr"'
                                 'be careful these langs are what you used for pretraining (the same order),'
                                 'not for finetuning.'
                                 'you should always add all pretraining language idx during finetuning.')
        # fmt: on

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args, src_dict, tgt_dict)
        self.langs = args.langs.split(',')
        # for d in [src_dict, tgt_dict]:
        #     for l in self.langs:
        #         d.add_symbol('[{}]'.format(l))
        #     d.add_symbol('<mask>')
        
        langmap = {lang.split("_")[0]: lang for lang in self.langs}
        logger.info(langmap)

        self.src_lang_tag = langmap.get(self.args.source_lang, "unk")
        self.tgt_lang_tag = langmap.get(self.args.target_lang, "unk")

    @classmethod
    def load_dictionary(cls, filename):
        d = Dictionary.load(filename)

        langs = "ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN"
        langs = langs.split(',')
        for l in langs:
            d.add_symbol('[{}]'.format(l))
        d.add_symbol('<mask>')
        
        return d

    def build_generator(self, models, args):
        print(self.tgt_dict.index('[{}]'.format(self.tgt_lang_tag)), len(self.tgt_dict))
        if getattr(args, 'score_reference', False):
            from fairseq.sequence_scorer import SequenceScorer
            return SequenceScorer(
                self.target_dictionary,
                eos=self.tgt_dict.index('[{}]'.format(self.tgt_lang_tag))
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
                eos=self.tgt_dict.index('[{}]'.format(self.tgt_lang_tag))
            )

    def build_dataset_for_inference(self, src_tokens, src_lengths):
        src_lang_id = self.source_dictionary.index('[{}]'.format(self.src_lang_tag))
        source_tokens = []
        for s_t in src_tokens:
            s_t = torch.cat([s_t, s_t.new(1).fill_(src_lang_id)])
            source_tokens.append(s_t)
        dataset = LanguagePairDataset(source_tokens, src_lengths, self.src_lang_tag)
        return dataset
