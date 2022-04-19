import torch.nn as nn
import torch


class LayerNormAnnotated(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNormAnnotated, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        return x


class LNorm(nn.Module):
    """A wraper to choose the type of normalization to be applied"""

    def __init__(self, config):

        super(LNorm, self).__init__()

        if config.layer_norm_type == "pytorch":
            self.internal = nn.LayerNorm(config.hidden_dim, eps=1e-12)
        elif config.layer_norm_type == "annotated":
            self.internal = LayerNormAnnotated(config.hidden_dim, eps=1e-6)
        else:
            raise Exception("{} no implemented".format(config.layer_norm_type))

    def forward(self, x):
        return self.internal(x)
