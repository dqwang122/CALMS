import torch
import torch.nn.functional as F

import math
from fairseq import metrics, utils
from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import (
    LabelSmoothedCrossEntropyCriterion,
    label_smoothed_nll_loss
)

@register_criterion('cross_entropy_and_ranking')
class CrossEntropyAndRanking(LabelSmoothedCrossEntropyCriterion):

    def __init__(self, task, ranking_head_name, sentence_avg, label_smoothing, ranking_loss_weight, ranking_loss_margin):
        super().__init__(task, sentence_avg, label_smoothing)
        self.ranking_head_name = ranking_head_name
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ranking_loss_weight = ranking_loss_weight
        self.ranking_loss_margin = ranking_loss_margin

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--ranking-head-name', type=str, default=None, help='use ranking during training')
        parser.add_argument('--ranking-loss-weight', type=int, default=-1, help='ranking loss during training')
        parser.add_argument('--ranking-loss-margin', type=float, default=1.0, help='ranking margin during training')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        assert (
            hasattr(model, 'classification_heads')
            and self.ranking_head_name in model.classification_heads
        ), 'model must provide sentence ranking head for --criterion=cross_entropy_and_ranking'

        weight = self.ranking_loss_weight if self.ranking_loss_weight != -1 else sample['ntokens']
        margin = self.ranking_loss_margin

        summ, extra = model(**sample['net_input'], classification_head_name=self.ranking_head_name)
        loss, nll_loss = self.compute_loss(model, tuple([summ]), sample['target'], reduce=reduce)
        rank_loss = self.compute_rank_loss(model, tuple([extra['margin_states']]), sample['label'], margin=margin,weight=weight, reduce=reduce)
        sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']
        # print(loss.data, rank_loss.data)
        loss = loss + rank_loss
        logging_output = {
            'loss': loss.data,
            'rank_loss': rank_loss.data,
            'nll_loss': nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, target, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = target.view(-1, 1)  # sample["target"]
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
        )
        return loss, nll_loss

    def compute_rank_loss(self, model, net_output, target, margin=1.0, weight=1, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = target.view(-1)  # sample["label"]
        # print(lprobs.size(), target.size())
        loss = F.multi_margin_loss(lprobs, target, margin=margin, reduce=reduce) * weight
        return loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        rank_loss_sum = sum(log.get('rank_loss', 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get('nll_loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('nll_loss', nll_loss_sum / ntokens / math.log(2), sample_size, round=3)
        metrics.log_scalar('rank_loss_sum', rank_loss_sum / sample_size / math.log(2), ntokens, round=3)
        metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))


