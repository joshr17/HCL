
import math

import torch
import torch.nn.functional as F

from cortex_DIM.functions.misc import log_sum_exp


def raise_measure_error(measure):
    supported_measures = ['GAN', 'JSD', 'JSD_hard', \
                            'X2', 'KL', 'RKL', 'DV', 'H2', 'W1']
    raise NotImplementedError(
        'Measure `{}` not supported. Supported: {}'.format(measure,
                                                           supported_measures))
    
def get_positive_expectation(p_samples, measure, average=True, tau_plus=0.5):
    """Computes the positive part of a divergence / difference.

    Args:
        p_samples: Positive samples.
        measure: Measure to compute for.
        average: Average the result over samples.

    Returns:
        torch.Tensor

    """
    log_2 = math.log(2.)

    if measure == 'GAN':
        Ep = - F.softplus(-p_samples)
    elif measure == 'JSD':
        Ep = log_2 - F.softplus(- p_samples)
    elif measure == 'JSD_hard':
        Ep = log_2 - F.softplus(- p_samples) - (tau_plus/(1-tau_plus)) *(F.softplus(-p_samples) + p_samples)
    elif measure == 'X2':
        Ep = p_samples ** 2
    elif measure == 'KL':
        Ep = p_samples + 1.
    elif measure == 'RKL':
        Ep = -torch.exp(-p_samples)
    elif measure == 'DV':
        Ep = p_samples
    elif measure == 'H2':
        Ep = 1. - torch.exp(-p_samples)
    elif measure == 'W1':
        Ep = p_samples
    else:
        raise_measure_error(measure)

    if average:
        return Ep.mean()
    else:
        return Ep


def get_negative_expectation(q_samples, measure, average=True, beta=0, tau_plus=0.5):
    """Computes the negative part of a divergence / difference.

    Args:
        q_samples: Negative samples.
        measure: Measure to compute for.
        average: Average the result over samples.

    Returns:
        torch.Tensor

    """
    log_2 = math.log(2.)

    if measure == 'GAN':
        Eq = F.softplus(-q_samples) + q_samples
    elif measure == 'JSD':
        Eq = F.softplus(-q_samples) + q_samples - log_2
    elif measure == 'JSD_hard':
        if beta==0:
            Eq = get_negative_expectation(q_samples, measure='JSD', average=average, beta=0)
            Eq=Eq/(1-tau_plus) 
        else:
            Eq = F.softplus(-q_samples) + q_samples 

            reweight= -2*q_samples / max( q_samples.max(), q_samples.min().abs())
            reweight=(beta * reweight).exp()
            reweight = reweight / reweight.mean(dim=1).view(-1,1)

            Eq = (reweight * Eq)
            Eq=Eq/(1-tau_plus) 
            Eq-=log_2
    elif measure == 'X2':
        Eq = -0.5 * ((torch.sqrt(q_samples ** 2) + 1.) ** 2)
    elif measure == 'KL':
        Eq = torch.exp(q_samples)
    elif measure == 'RKL':
        Eq = q_samples - 1.
    elif measure == 'DV':
        Eq = log_sum_exp(q_samples, 0) - math.log(q_samples.size(0))
    elif measure == 'H2':
        Eq = torch.exp(q_samples) - 1.
    elif measure == 'W1':
        Eq = q_samples
    else:
        raise_measure_error(measure)

    if average:
        return Eq.mean()
    else:
        return Eq