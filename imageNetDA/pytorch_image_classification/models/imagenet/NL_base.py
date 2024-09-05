"""Common utilities for mobilenet."""
import abc
import collections
import logging
import functools
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class ZeroInitBN(nn.BatchNorm2d):
    """BatchNorm with zero initialization."""

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            nn.init.zeros_(self.weight)
            nn.init.zeros_(self.bias)


class Nonlocal(nn.Module):
    """Lightweight Non-Local Module.
    See https://arxiv.org/abs/2004.01961
    """

    def __init__(self, n_feature, nl_c, nl_s, norm_method='batch_norm', batch_norm_kwargs=None):
        super(Nonlocal, self).__init__()
        self.n_feature = n_feature
        self.nl_c = nl_c
        self.nl_s = nl_s

        self.depthwise_conv = nn.Conv2d(n_feature,
                                        n_feature,
                                        3,
                                        1,
                                        (3 - 1) // 2,
                                        groups=n_feature,
                                        bias=False)

        if batch_norm_kwargs is None:
            batch_norm_kwargs = {}
        # from modeling.backbones.searched_mobilenetv2.utils.config import FLAGS
        # if hasattr(FLAGS, 'nl_norm'):  # TODO: as param
        #     self.bn = get_nl_norm_fn(FLAGS.nl_norm)(n_feature, **batch_norm_kwargs)
        # else:
        #     self.bn = ZeroInitBN(n_feature, **batch_norm_kwargs)
        if norm_method == 'instance_norm':
            self.bn = functools.partial(nn.InstanceNorm2d, affine=True, track_running_stats=True)(
                n_feature,
                **batch_norm_kwargs
            )
        if norm_method == 'batch_norm':
            self.bn = ZeroInitBN(n_feature, **batch_norm_kwargs)

    def forward(self, l):

        N, n_in, H, W = list(l.shape)

        reduced_HW = (H // self.nl_s) * (W // self.nl_s)

        l_reduced = l[:, :, ::self.nl_s, ::self.nl_s]

        theta, phi, g = l[:, :int(self.nl_c * n_in), :, :], l_reduced[:, :int(self.nl_c * n_in), :, :], l_reduced

        # Employing associative law of matrix multiplication
        if (H * W) * reduced_HW * n_in * (1 + self.nl_c) \
                < \
                (H * W) * n_in**2 * self.nl_c + reduced_HW * n_in**2 * self.nl_c:

            f = torch.einsum('niab,nicd->nabcd', theta, phi)
            f = torch.einsum('nabcd,nicd->niab', f, g)

        else:

            f = torch.einsum('nihw,njhw->nij', phi, g)
            f = torch.einsum('nij,nihw->njhw', f, theta)

        f = f / H * W

        f = self.bn(self.depthwise_conv(f))

        return f + l

    def __repr__(self):
        return '{}({}, nl_c={}, nl_s={}'.format(self._get_name(),
                                                self.n_feature, self.nl_c,
                                                self.nl_s)

################## non-local block #########################################

