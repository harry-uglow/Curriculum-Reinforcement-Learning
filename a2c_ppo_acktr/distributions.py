import torch
import torch.nn as nn
import torch.nn.functional as F

from a2c_ppo_acktr.utils import AddBias, init

"""
Modify standard PyTorch distributions so they are compatible with this code.
"""

#
# Standardize distribution interfaces
#

# Categorical
FixedCategorical = torch.distributions.Categorical

old_sample = FixedCategorical.sample
FixedCategorical.sample = lambda self: old_sample(self).unsqueeze(-1)

log_prob_cat = FixedCategorical.log_prob
FixedCategorical.log_probs = lambda self, actions: log_prob_cat(self, actions.squeeze(-1)).view(
    actions.size(0), -1).sum(-1).unsqueeze(-1)

FixedCategorical.mode = lambda self: self.probs.argmax(dim=-1, keepdim=True)

# Normal
FixedNormal = torch.distributions.Normal

log_prob_normal = FixedNormal.log_prob
FixedNormal.log_probs = lambda self, actions: log_prob_normal(self, actions).sum(-1, keepdim=True)

normal_entropy = FixedNormal.entropy
FixedNormal.entropy = lambda self: normal_entropy(self).sum(-1)

FixedNormal.mode = lambda self: self.mean

# Bernoulli
FixedBernoulli = torch.distributions.Bernoulli

log_prob_bernoulli = FixedBernoulli.log_prob
FixedBernoulli.log_probs = lambda self, actions: log_prob_bernoulli(self, actions).view(
    actions.size(0), -1).sum(-1).unsqueeze(-1)

bernoulli_entropy = FixedBernoulli.entropy
FixedBernoulli.entropy = lambda self: bernoulli_entropy(self).sum(-1)
FixedBernoulli.mode = lambda self: torch.gt(self.probs, 0.5).float()

# Beta
FixedBeta = torch.distributions.Beta

log_prob_beta = FixedBeta.log_prob
FixedBeta.log_probs = lambda self, actions: log_prob_beta(self, actions).sum(-1, keepdim=True)

beta_entropy = FixedBeta.entropy
FixedBeta.entropy = lambda self: beta_entropy(self).sum(-1)

FixedBeta.mode = lambda self: self.mean


class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()

        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               gain=0.01)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedCategorical(logits=x)


class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs, zll=False):
        super(DiagGaussian, self).__init__()

        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0))

        init_zeros = lambda m: init(m, lambda x, **kwargs: nn.init.constant_(x, 0),
                                    lambda x: nn.init.constant_(x, 0))

        init_last_layer = init_zeros if zll else init_

        self.fc_mean = init_last_layer(nn.Linear(num_inputs, num_outputs))
        self.logstd = AddBias(torch.zeros(num_outputs))

    def forward(self, x):
        action_mean = self.fc_mean(x)

        #  An ugly hack for my KFAC implementation.
        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.cuda()

        action_logstd = self.logstd(zeros)
        return FixedNormal(action_mean, action_logstd.exp())


class Bernoulli(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Bernoulli, self).__init__()

        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0))

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedBernoulli(logits=x)


class Beta(nn.Module):
    """
    Added beta distribution after reading
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/issues/109.
    Not used in final environment architecture
    """
    def __init__(self, num_inputs, num_outputs, zll=False):
        super(Beta, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))

        init_zeros = lambda m: init(m, lambda x, **kwargs: nn.init.constant_(x, 0),
                                    lambda x: nn.init.constant_(x, 0))

        init_last_layer = init_zeros if zll else init_

        self.alpha_linear = init_last_layer(nn.Linear(num_inputs, num_outputs))
        self.beta_linear = init_last_layer(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        alpha = F.softplus(self.alpha_linear(x)) + 1
        beta = F.softplus(self.beta_linear(x)) + 1

        return FixedBernoulli(logits=x)
