import torch
from torch.nn import functional as F

def get_composite_loss(losses, weights):
    '''
    get_composite_loss() returns a composite loss that is a weighted linear
    combination of other losses defined here. This function returns another
    function, which itself is the combined / composite loss.

    The arguments it accepts are:
        - losses (list of func): a list of loss functions with the traditional
          loss function signature.
        - weights (list of float): a list of floats, one per loss function,
          that are scalar multipliers to the output of eadh loss function.

    The values it returns are:
        - loss_fn (func): the loss function consisting of a linear combination
          of the given losses with the given weights. It accepts all the same
          arguments as a normal loss function in this file. Note that the same
          kwargs are passed to all atomic losses, so (for designers / coders),
          make sure that the names of distinct hyperparamters never overlap to
          avoid collisions!
    '''
    assert len(losses) == len(weights), \
      f'losses and weights must match 1:1 but are {len(losses)} : {len(weights)}'
    
    def loss_fn(score_pos, score_neg, reduction='mean', **kwargs):
        loss = 0
        for i in range(len(losses)):
            loss += weights[i] * losses[i](
                score_pos,
                score_neg,
                reduction=reduction,
                **kwargs
            )
        return loss
    
    return loss_fn

def pairwise_logistic_loss(score_pos, score_neg, reduction='mean', **kwargs):
    '''
    pairwise_logistic_loss() is an implementation of Pairwise Logistic Loss for
    TWIG. It is calculated as log(1 + exp((score_neg - score_pos))

    The arguments it accepts are:
        - score_pos (Tensor): the scores of all positive triples. Note when
          npp > 1, score_pos MUST have its rows expanded such that each
          negative score in score_neg at index i has, at index i in score_pos,
          the score of the positive triple from which it comes. This means that
          score_pos will have repeated elements.
        - score_neg (Tensor) the scores of all negative triples, with indicies
          the the same order as, and corresponding to, those in score_pos.
        - reduction (str): 'mean' or 'sum', the method why which loss values
          are reduced from tensors into single scalars. Default 'mean'.
        - **kwargs (dict str -> value): a dict containing any other arguments
          or hyperparameters needed for specific loss function with the
          argument as the string and the value it corresponds to as the value.
          See "Special arguments accepted" above for details on such arguments.

    The values it returns are:
        - loss (Tensor): the calculated loss value as a single float scalar.
    '''
    if reduction == 'mean':
        reduce = torch.mean
    elif reduction == 'sum':
        reduce = torch.sum
    else:
        assert False, f"invalid reduction: {reduction}"

    loss = reduce(
        torch.log(1 + torch.exp((score_neg - score_pos)))
    )
    return loss

def mse_label_loss(score_pos, score_neg, reduction='mean', **kwargs):
    '''
    mse_label_loss() is an implementation of Mean Squared Error Loss for TWIG.
    It calculates the mean squared error of model predictions from 1 (true) and
    0 (false).

    Special arguments accepted:
        - "npp" -> int, the number of negative generated per positive triple.
          If given, it will scale down the loss given to positive triple
          proportional to npp such that all triples, positive or negative, are
          given equal weight. If this rescaling is not wanted, set npp to -1.

    The arguments it accepts are:
        - score_pos (Tensor): the scores of all positive triples. Note when
          npp > 1, score_pos MUST have its rows expanded such that each
          negative score in score_neg at index i has, at index i in score_pos,
          the score of the positive triple from which it comes. This means that
          score_pos will have repeated elements.
        - score_neg (Tensor) the scores of all negative triples, with indicies
          the the same order as, and corresponding to, those in score_pos.
        - reduction (str): 'mean' or 'sum', the method why which loss values
          are reduced from tensors into single scalars. Default 'mean'.
        - **kwargs (dict str -> value): a dict containing any other arguments
          or hyperparameters needed for specific loss function with the
          argument as the string and the value it corresponds to as the value.
          See "Special arguments accepted" above for details on such arguments.

    The values it returns are:
        - loss (Tensor): the calculated loss value as a single float scalar.
    '''
    if reduction == 'mean':
        reduce = torch.mean
    elif reduction == 'sum':
        reduce = torch.sum
    else:
        assert False, f"invalid reduction: {reduction}"

    npp = torch.max(kwargs["npp"], 1) #if npp is -1 just use 1
    true_label = torch.ones(score_pos.shape)
    false_label = torch.ones(score_neg.shape)
    loss_pos = F.mse_loss(score_pos, true_label, reduction=reduction) / npp
    loss_neg = F.mse_loss(score_neg, false_label, reduction=reduction)
    loss = reduce(
        loss_pos + loss_neg
    )
    return loss
    
def margin_ranking_loss(score_pos, score_neg, reduction='mean', **kwargs):
    '''
    margin_ranking_loss() is an implementation of Margin Ranking Loss for TWIG.
    It is calculated as max((score_neg - score_pos + margin), 0).

    Special arguments accepted:
        - "margin" -> float, the margin value to use when computing margin
          ranking loss. This represents the desired margine the model should
          enforce between the values of the scores of positive and negative
          triples.

    The arguments it accepts are:
        - score_pos (Tensor): the scores of all positive triples. Note when
          npp > 1, score_pos MUST have its rows expanded such that each
          negative score in score_neg at index i has, at index i in score_pos,
          the score of the positive triple from which it comes. This means that
          score_pos will have repeated elements.
        - score_neg (Tensor) the scores of all negative triples, with indicies
          the the same order as, and corresponding to, those in score_pos.
        - reduction (str): 'mean' or 'sum', the method why which loss values
          are reduced from tensors into single scalars. Default 'mean'.
        - **kwargs (dict str -> value): a dict containing any other arguments
          or hyperparameters needed for specific loss function with the
          argument as the string and the value it corresponds to as the value.
          See "Special arguments accepted" above for details on such arguments.

    The values it returns are:
        - loss (Tensor): the calculated loss value as a single float scalar.
    '''
    if reduction == 'mean':
        reduce = torch.mean
    elif reduction == 'sum':
        reduce = torch.sum
    else:
        assert False, f"invalid reduction: {reduction}"

    margin = kwargs['margin']
    loss = reduce(
        (score_neg - score_pos + margin).relu()
    )
    return loss
