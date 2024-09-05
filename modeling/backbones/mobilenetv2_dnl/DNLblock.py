import abc
import collections
import logging
import functools
import math
import numpy as np
import torch
from torch import nn
import types
from torch.nn import functional as F


class ZeroInitBN(nn.BatchNorm2d):
    """BatchNorm with zero initialization."""

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            nn.init.zeros_(self.weight)
            nn.init.zeros_(self.bias)


class DNonlocal(nn.Module):

    def __init__(self, n_feature, initial_nl_c, nl_s, norm_method='batch_norm', batch_norm_kwargs=None):
        super(DNonlocal, self).__init__()
        self.n_feature = n_feature
        self.initial_nl_c = initial_nl_c
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

        if norm_method == 'instance_norm':
            self.bn = functools.partial(nn.InstanceNorm2d, affine=True, track_running_stats=True)(
                n_feature,
                **batch_norm_kwargs
            )
        if norm_method == 'batch_norm':
            self.bn = ZeroInitBN(n_feature, **batch_norm_kwargs)

        startpos_int = torch.from_numpy(np.array([0]))
        self.ch_startpos_int = torch.nn.Parameter(startpos_int.to(torch.int64), requires_grad=False)
        # self.ch_startpos_int = np.array([0])
        self.ch_startpos_dec = torch.nn.Parameter(torch.Tensor([0.0]), requires_grad=True)
        # nl_c * n_feature is the initial size
        length_int = torch.from_numpy(np.array([int(self.initial_nl_c * n_feature)]))
        self.ch_length_int = torch.nn.Parameter(length_int.to(torch.int64), requires_grad=False)
        self.ch_length_dec = torch.nn.Parameter(torch.Tensor([0.0]), requires_grad=True)

    def forward(self, l):

        N, channel_in, H, W = list(l.shape)

        reduced_HW = (H // self.nl_s) * (W // self.nl_s)

        l_reduced = l[:, :, ::self.nl_s, ::self.nl_s]

        # self.window_length = int(self.nl_c * channel_in)

        # if not isinstance(self.ch_startpos_int, torch.nn.Parameter):
        #     self.ch_startpos_int = torch.from_numpy(self.ch_startpos_int)
        #     self.ch_startpos_int = torch.nn.Parameter(self.ch_startpos_int.to(torch.int8), requires_grad=False)

        g = l_reduced

        l_rep = l
        l_padded = torch.cat((l, l_rep), dim=1)

        lr_rep = l_reduced
        lr_padded = torch.cat((l_reduced, lr_rep), dim=1)

        startpos_int = self.ch_startpos_int

        length_int = self.ch_length_int

        theta_1 = l_padded[:, startpos_int:startpos_int + length_int, :, :]

        theta_2 = l_padded[:, startpos_int + 1:startpos_int + length_int + 1, :, :]

        phi_1 = lr_padded[:, startpos_int:startpos_int + length_int, :, :]

        phi_2 = lr_padded[:, startpos_int + 1:startpos_int + length_int + 1, :, :]

        theta_s = theta_1 * (1 - self.ch_startpos_dec) + theta_2 * self.ch_startpos_dec
        # theta_n, theta_ch, theta_h, theta_w = list(theta_s.shape)
        # theta_pad_zero = torch.nn.Parameter(torch.zeros(theta_n, 1, theta_h, theta_w), requires_grad=False)
        # theta_pad_zero = theta_pad_zero.cuda()
        # theta_s_padded = torch.cat((theta_s, theta_pad_zero), dim=1)

        phi_s = phi_1 * (1 - self.ch_startpos_dec) + phi_2 * self.ch_startpos_dec


        if (H * W) * reduced_HW * channel_in * (1 + (self.ch_length_int // self.n_feature)) \
                < \
                (H * W) * channel_in**2 * (self.ch_length_int // self.n_feature) + reduced_HW * channel_in**2 * (self.ch_length_int // self.n_feature):

            f_s = torch.einsum('niab,nicd->nabcd', theta_s, phi_s)
            f_s = torch.einsum('nabcd,nicd->niab', f_s, g)

        else:

            f_s = torch.einsum('nihw,njhw->nij', phi_s, g)
            f_s = torch.einsum('nij,nihw->njhw', f_s, theta_s)

        f_s = f_s // H * W

        # phi_n, phi_ch, phi_h, phi_w = list(phi_s.shape)
        # phi_pad_zero = torch.nn.Parameter(torch.zeros(phi_n, 1, phi_h, phi_w), requires_grad=False)
        # phi_pad_zero = phi_pad_zero.cuda()
        # phi_s_padded = torch.cat((phi_s, phi_pad_zero), dim=1)

        theta_1_l = l_padded[:, startpos_int:startpos_int + length_int + 1, :, :]

        theta_2_l = l_padded[:, startpos_int + 1:startpos_int + length_int + 1 + 1, :, :]

        phi_1_l = lr_padded[:, startpos_int:startpos_int + length_int + 1, :, :]

        phi_2_l = lr_padded[:, startpos_int + 1:startpos_int + length_int + 1 + 1, :, :]

        theta_l = theta_1_l * (1 - self.ch_startpos_dec) + theta_2_l * self.ch_startpos_dec
        phi_l = phi_1_l * (1 - self.ch_startpos_dec) + phi_2_l * self.ch_startpos_dec

        # theta = theta_s_padded * (1 - self.ch_length_dec) + theta_l * self.ch_length_dec
        # phi = phi_s_padded * (1 - self.ch_length_dec) + phi_l * self.ch_length_dec

        if (H * W) * reduced_HW * channel_in * (1 + (self.ch_length_int // self.n_feature)) \
                < \
                (H * W) * channel_in**2 * (self.ch_length_int // self.n_feature) + reduced_HW * channel_in**2 * (self.ch_length_int // self.n_feature):

            f_l = torch.einsum('niab,nicd->nabcd', theta_l, phi_l)
            f_l = torch.einsum('nabcd,nicd->niab', f_l, g)

        else:

            f_l = torch.einsum('nihw,njhw->nij', phi_l, g)
            f_l = torch.einsum('nij,nihw->njhw', f_l, theta_l)

        f_l = f_l // H * W

        f = f_s * (1 - self.ch_length_dec) + f_l * self.ch_length_dec

        f = self.bn(self.depthwise_conv(f))

        return f + l

    def __repr__(self):
        return '{}({}, nl_c={}, nl_s={}'.format(self._get_name(),
                                                self.n_feature, (self.ch_length_int // self.n_feature),
                                                (self.ch_length_int // self.n_feature))

    # def _set_statrtpos_int(self, new_int):
    #     self.ch_startpos_int = new_int

    # def _set_ch_startpos_int(self, ch_startpos_int):
    #     self.ch_startpos_int = ch_startpos_int

