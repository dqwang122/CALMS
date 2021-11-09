# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
BART: Denoising Sequence-to-Sequence Pre-training for
Natural Language Generation, Translation, and Comprehension
"""

import logging

import torch.nn as nn

from fairseq import utils
from fairseq.models import (
    register_model,
    register_model_architecture,
)

from fairseq.data import data_utils

from fairseq.models.fairseq_model import BaseFairseqModel
from fairseq.models.bart.model import BARTModel, mbart_large_architecture
from fairseq.models.transformer import base_architecture
from fairseq.models.bart import BARTClassificationHead
from fairseq.modules.transformer_sentence_encoder import init_bert_params

logger = logging.getLogger(__name__)

DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024


@register_model('rank_summ')
class RankSumm(BARTModel):

    def __init__(self, args, encoder, decoder):

        super().__init__(args, encoder, decoder)

    @staticmethod
    def add_args(parser):
        super(RankSumm, RankSumm).add_args(parser)

    def forward(
        self, src_tokens, src_lengths, margin, prev_output_tokens=None,
        features_only=False, classification_head_name=None, **kwargs
    ):

        x, extra = BARTModel.forward(self, src_tokens, src_lengths,
                                    prev_output_tokens,features_only=False,
                                    classification_head_name=None, **kwargs)

        # extra: {"attn": [attn], "inner_states": inner_states, 'encoder_out': encoder_out}
        # inner_states (1 + layer_number) * [tgt__len, bs, hidden_sizes]
        # encoder_out: [src_len, bs, hidden_sizes]

        # margin [bs, num, seq_len] , '1' means mask

        seq_len, bs, hidden_sizes = extra['encoder_out'].size()
        num = margin.size(1)
        # print(extra['encoder_out'].size(), margin.size())

        if classification_head_name is not None:
            document_states = extra['encoder_out'].permute(1, 0, 2) # [bs, seqlen, hidden_sizes]
            document_states = document_states.unsqueeze(1).repeat(1,num,1,1)    # [bs, num, seqlen, hidden_sizes]
            document_states = document_states.contiguous().view(-1, seq_len, hidden_sizes)

            margin = margin.contiguous().view(-1, seq_len)   # [bs * num, seqlen]
            margin_states = document_states.masked_fill_(margin.unsqueeze(-1), 0)    # [bs * num, seqlen, hidden_sizes]
            margin_states = margin_states.sum(axis=1) # [bs * num, hidden_state]

            margin_states =  margin_states.contiguous().view(bs, -1, hidden_sizes) # [bs, num, hidden_state]
            extra['margin_states'] = self.classification_heads[classification_head_name](margin_states).squeeze(-1) # [bs, num]

        return x, extra



@register_model_architecture('rank_summ', 'rank_summ_large')
def rank_summ_large_architecture(args):
    mbart_large_architecture(args)
