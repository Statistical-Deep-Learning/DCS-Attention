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

from modeling.backbones.searched_mobilenetv2.utils.common import add_prefix


class Bottleneck(nn.Module):

    def __init__(self, in_channels, out_channels, expansion_factor, stride=1):

        super(Bottleneck, self).__init__()

        mid_channels = in_channels * expansion_factor

        self.use_residual = stride == 1 and in_channels == out_channels

        self.conv1 = ConvBlock(in_channels, mid_channels, 1)

        self.dwconv2 = ConvBlock(mid_channels, mid_channels, 3, stride, 1, g=mid_channels)

        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):

        m = self.conv1(x)

        m = self.dwconv2(m)

        m = self.conv3(m)

        if self.use_residual:
            return x + m

        else:
            return m

def _make_divisible(v, divisor, min_value=None):
    """Make channels divisible to divisor.
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class CheckpointModule(nn.Module, metaclass=abc.ABCMeta):
    """Discard mid-result using checkpoint."""

    def __init__(self, use_checkpoint=True):
        super(CheckpointModule, self).__init__()
        self._use_checkpoint = use_checkpoint

    def forward(self, *args, **kwargs):
        from torch.utils.checkpoint import checkpoint
        if self._use_checkpoint:
            return checkpoint(self._forward, *args, **kwargs)
        return self._forward(*args, **kwargs)

    @abc.abstractmethod
    def _forward(self, *args, **kwargs):
        pass


class Identity(nn.Module):
    """Module proxy for null op."""

    def forward(self, x):
        return x


class Narrow(nn.Module):
    """Module proxy for `torch.narrow`."""

    def __init__(self, dimension, start, length):
        super(Narrow, self).__init__()
        self.dimension = dimension
        self.start = start
        self.length = length

    def forward(self, x):
        return x.narrow(self.dimension, self.start, self.length)


class Swish(nn.Module):
    """Swish activation function.
    See: https://arxiv.org/abs/1710.05941
    NOTE: Will consume much more GPU memory compared with inplaced ReLU.
    """

    def forward(self, x):
        return x * torch.sigmoid(x)


class HSwish(object):
    """Hard Swish activation function.
    See: https://arxiv.org/abs/1905.02244
    """

    def forward(self, x):
        return x * F.relu6(x + 3, True).div_(6)


class SqueezeAndExcitation(nn.Module):
    """Squeeze-and-Excitation module.
    See: https://arxiv.org/abs/1709.01507
    """

    def __init__(self,
                 n_feature,
                 n_hidden,
                 spatial_dims=[2, 3],
                 active_fn=None):
        super(SqueezeAndExcitation, self).__init__()
        self.n_feature = n_feature
        self.n_hidden = n_hidden
        self.spatial_dims = spatial_dims
        self.se_reduce = nn.Conv2d(n_feature, n_hidden, 1, bias=True)
        self.se_expand = nn.Conv2d(n_hidden, n_feature, 1, bias=True)
        self.active_fn = active_fn()

    def forward(self, x):
        se_tensor = x.mean(self.spatial_dims, keepdim=True)
        se_tensor = self.se_expand(self.active_fn(self.se_reduce(se_tensor)))
        return torch.sigmoid(se_tensor) * x

    def __repr__(self):
        return '{}({}, {}, spatial_dims={}, active_fn={})'.format(
            self._get_name(), self.n_feature, self.n_hidden, self.spatial_dims,
            self.active_fn)


class ZeroInitBN(nn.BatchNorm2d):
    """BatchNorm with zero initialization."""

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            nn.init.zeros_(self.weight)
            nn.init.zeros_(self.bias)

################## Bottle non-local block #########################################
class Bottle_Nonlocal(nn.Module):
    """Lightweight Non-Local Module.
    See https://arxiv.org/abs/2004.01961
    """

    def __init__(self, n_feature, nl_c, nl_s, norm_method='batch_norm', batch_norm_kwargs=None):
        super(Bottle_Nonlocal, self).__init__()
        self.n_feature = n_feature
        self.nl_c = nl_c
        self.nl_s = nl_s

        self.bottle = Bottleneck(n_feature, n_feature, expansion_factor=6, stride=1)

        if batch_norm_kwargs is None:
            batch_norm_kwargs = {}
        # from modeling.backbones.searched_mobilenetv2.utils.config import FLAGS
        # if hasattr(FLAGS, 'nl_norm'):  # TODO: as param
        #     self.bn = get_nl_norm_fn(FLAGS.nl_norm)(n_feature, **batch_norm_kwargs)
        # else:
        #     self.bn = ZeroInitBN(n_feature, **batch_norm_kwargs)
        # if norm_method == 'instance_norm':
        #     self.bn = functools.partial(nn.InstanceNorm2d, affine=True, track_running_stats=True)(
        #         n_feature,
        #         **batch_norm_kwargs
        #     )
        # if norm_method == 'batch_norm':
        #
        # self.bn = ZeroInitBN(n_feature, **batch_norm_kwargs)

    def forward(self, l):

        l = l

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

        f = self.bottle(f)

        return f + l


################## non-local block #########################################

class ChannelNonlocal(nn.Module):
    """Lightweight Non-Local Module.
    See https://arxiv.org/abs/2004.01961
    """

    def __init__(self, n_feature, nl_c, nl_s, norm_method='batch_norm', batch_norm_kwargs=None):
        super(ChannelNonlocal, self).__init__()
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

        f = torch.einsum('niab,njab->nij', g, phi)
        f = torch.einsum('nij,njhw->nihw', f, theta)

        f = f / n_in

        f = self.bn(self.depthwise_conv(f))

        return f + l

    def __repr__(self):
        return '{}({}, nl_c={}, nl_s={}'.format(self._get_name(),
                                                self.n_feature, self.nl_c,
                                                self.nl_s)

class ConvBlock(nn.Module):
    """Basic convolutional block.

    convolution (bias discarded) + batch normalization + relu6.

    Args:
        in_c (int): number of input channels.
        out_c (int): number of output channels.
        k (int or tuple): kernel size.
        s (int or tuple): stride.
        p (int or tuple): padding.
        g (int): number of blocked connections from input channels
            to output channels (default: 1).
    """

    def __init__(self, in_c, out_c, k, s=1, p=0, g=1):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(
            in_c, out_c, k, stride=s, padding=p, bias=False, groups=g
        )

        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return F.relu6(self.bn(self.conv(x)))

class MyNonlocal(nn.Module):
    """Lightweight Non-Local Module.
    See https://arxiv.org/abs/2004.01961
    """

    def __init__(self, n_feature, nl_c, nl_s, norm_method='batch_norm', batch_norm_kwargs=None):
        super(MyNonlocal, self).__init__()
        self.n_feature = n_feature
        self.nl_c = nl_c
        self.nl_s = nl_s

        self.conv_inside = ConvBlock(n_feature, n_feature, 1)

        self.depthwise_conv = nn.Conv2d(n_feature,
                                        n_feature,
                                        3,
                                        1,
                                        (3 - 1) // 2,
                                        groups=n_feature,
                                        bias=False)
        self.softmax_1 = nn.Softmax(dim=1)
        self.softmax_2 = nn.Softmax(dim=2)
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

        l_reduced = l[:, :, :, :]
        l_reduced_afterconv = self.conv_inside(l_reduced)

        theta, phi, g = l[:, :int(self.nl_c * n_in), :, :], l[:, :int(self.nl_c * n_in), :, :], l_reduced_afterconv

        # Employing associative law of matrix multiplication


        f = torch.einsum('niab,nicd->nabcd', theta, phi)
        original_size = f.size()

        f = torch.flatten(f, start_dim=1, end_dim=2)
        f = self.softmax_1(f)
        f = torch.flatten(f, start_dim=2, end_dim=3)
        f = self.softmax_2(f)
        f = f. view(original_size)


        f = torch.einsum('nabcd,nicd->niab', f, g)


        f = f / H * W

        f = self.bn(self.depthwise_conv(f))

        return f + l

    def __repr__(self):
        return '{}({}, nl_c={}, nl_s={}'.format(self._get_name(),
                                                self.n_feature, self.nl_c,
                                                self.nl_s)




class Nonlocal_Linear(nn.Module):
    """Lightweight Non-Local Module.
    See https://arxiv.org/abs/2004.01961
    """

    def __init__(self, n_feature, nl_c, nl_s, hw, norm_method='batch_norm', batch_norm_kwargs=None):
        super(Nonlocal_Linear, self).__init__()
        self.n_feature = n_feature
        self.nl_c = nl_c
        self.nl_s = nl_s

        self.linear_inside = nn.Linear(hw, hw, bias=False)

        self.depthwise_conv = nn.Conv2d(n_feature,
                                        n_feature,
                                        3,
                                        1,
                                        (3 - 1) // 2,
                                        groups=n_feature,
                                        bias=False)
        self.softmax_1 = nn.Softmax(dim=1)
        self.softmax_2 = nn.Softmax(dim=2)
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

        l_reduced = l[:, :, :, :]
        l_size = l_reduced.size()
        l_reduced = torch.flatten(l_reduced, start_dim=2, end_dim=3)
        l_reduced_afterconv = self.linear_inside(l_reduced)
        l_reduced_afterconv = l_reduced_afterconv.view(l_size)

        theta, phi, g = l[:, :int(self.nl_c * n_in), :, :], l[:, :int(self.nl_c * n_in), :, :], l_reduced_afterconv

        # Employing associative law of matrix multiplication


        f = torch.einsum('niab,nicd->nabcd', theta, phi)
        original_size = f.size()

        f = torch.flatten(f, start_dim=1, end_dim=2)
        f = self.softmax_1(f)
        f = torch.flatten(f, start_dim=2, end_dim=3)
        f = self.softmax_2(f)
        f = f. view(original_size)


        f = torch.einsum('nabcd,nicd->niab', f, g)


        f = f / H * W

        f = self.bn(self.depthwise_conv(f))

        return f + l

    def __repr__(self):
        return '{}({}, nl_c={}, nl_s={}'.format(self._get_name(),
                                                self.n_feature, self.nl_c,
                                                self.nl_s)




################## non-local block #########################################

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


class ConvBNReLU(nn.Sequential):
    """Convolution-BatchNormalization-ActivateFn."""

    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel_size=3,
                 stride=1,
                 groups=1,
                 active_fn=None,
                 batch_norm_kwargs=None):
        if batch_norm_kwargs is None:
            batch_norm_kwargs = {}
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes,
                      out_planes,
                      kernel_size,
                      stride,
                      padding,
                      groups=groups,
                      bias=False),
            nn.BatchNorm2d(out_planes, **batch_norm_kwargs), active_fn())


class InvertedResidualChannelsFused(nn.Module):
    """Speedup version of `InvertedResidualChannels` by fusing small kernels.
    NOTE: It may consume more GPU memory.
    Support `Squeeze-and-Excitation`.
    """

    def __init__(self,
                 inp,
                 oup,
                 stride,
                 channels,
                 kernel_sizes,
                 expand,
                 active_fn=None,
                 batch_norm_kwargs=None,
                 se_ratio=None,
                 nl_c=0,
                 nl_s=0):
        super(InvertedResidualChannelsFused, self).__init__()
        assert stride in [1, 2]
        assert len(channels) == len(kernel_sizes)

        self.input_dim = inp
        self.output_dim = oup
        self.expand = expand
        self.stride = stride
        self.kernel_sizes = kernel_sizes
        self.channels = channels
        self.use_res_connect = self.stride == 1 and inp == oup
        self.batch_norm_kwargs = batch_norm_kwargs
        self.active_fn = active_fn
        self.se_ratio = se_ratio
        self.nl_c = nl_c
        self.nl_s = nl_s

        (self.expand_conv, self.depth_ops, self.project_conv, self.se_op, self.nl_op) \
            = self._build(channels, kernel_sizes, expand, se_ratio, nl_c, nl_s)

    def _build(self, hidden_dims, kernel_sizes, expand, se_ratio, nl_c, nl_s):

        _batch_norm_kwargs = self.batch_norm_kwargs \
            if self.batch_norm_kwargs is not None else {}

        hidden_dim_total = sum(hidden_dims)

        if self.expand:
            # pw
            expand_conv = ConvBNReLU(self.input_dim,
                                     hidden_dim_total,
                                     kernel_size=1,
                                     batch_norm_kwargs=_batch_norm_kwargs,
                                     active_fn=self.active_fn)
        else:
            expand_conv = Identity()

        narrow_start = 0
        depth_ops = nn.ModuleList()

        for k, hidden_dim in zip(kernel_sizes, hidden_dims):
            layers = []
            # if expand:
            #     layers.append(Narrow(1, narrow_start, hidden_dim))
            #     narrow_start += hidden_dim
            # else:
            #     if hidden_dim != self.input_dim:
            #         raise RuntimeError('uncomment this for search_first model')
            #     logging.warning(
            #         'uncomment this for previous trained search_first model')
            layers.extend([
                # dw
                ConvBNReLU(hidden_dim,
                           hidden_dim,
                           kernel_size=k,
                           stride=self.stride,
                           groups=hidden_dim,
                           batch_norm_kwargs=_batch_norm_kwargs,
                           active_fn=self.active_fn),
            ])
            depth_ops.append(nn.Sequential(*layers))

        project_conv = nn.Sequential(
            nn.Conv2d(hidden_dim_total, self.output_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.output_dim, **_batch_norm_kwargs))

        # if expand and narrow_start != hidden_dim_total:
        #     raise ValueError('Part of expanded are not used')

        if se_ratio is not None and se_ratio > 0:
            se_op = SqueezeAndExcitation(hidden_dim_total,
                                         int(round(self.input_dim * se_ratio)),
                                         active_fn=self.active_fn)
        else:
            se_op = Identity()




################## non-local block #########################################

        if nl_c > 0:
            nl_op = Nonlocal(self.output_dim, nl_c, nl_s, batch_norm_kwargs=_batch_norm_kwargs)
        else:
            nl_op = Identity()


################## non-local block #########################################

        return expand_conv, depth_ops, project_conv, se_op, nl_op

    def get_depthwise_bn(self):
        """Get `[module]` list of BN after depthwise convolution."""
        return list(self.get_named_depthwise_bn().values())

    def get_named_depthwise_bn(self, prefix=None):
        """Get `{name: module}` pairs of BN after depthwise convolution."""
        res = collections.OrderedDict()
        for i, op in enumerate(self.depth_ops):
            children = list(op.children())
            if self.expand:
                idx_op = 1
            else:
                raise RuntimeError('Not search_first')
            conv_bn_relu = children[idx_op]
            assert isinstance(conv_bn_relu, ConvBNReLU)
            conv_bn_relu = list(conv_bn_relu.children())
            _, bn, _ = conv_bn_relu
            assert isinstance(bn, nn.BatchNorm2d)
            name = 'depth_ops.{}.{}.1'.format(i, idx_op)
            name = add_prefix(name, prefix)
            res[name] = bn
        return res

    def forward(self, x):
        res = self.expand_conv(x)
        res = [op(res) for op in self.depth_ops]
        if len(res) != 1:
            res = torch.cat(res, dim=1)
        else:
            res = res[0]
        res = self.se_op(res)
        res = self.project_conv(res)
        res = self.nl_op(res)
        if self.use_res_connect:
            return x + res
        return res

    def __repr__(self):
        return ('{}({}, {}, channels={}, kernel_sizes={}, expand={}, stride={},'
                ' se_ratio={}, nl_s={}, nl_c={})').format(
                    self._get_name(), self.input_dim, self.output_dim,
                    self.channels, self.kernel_sizes, self.expand, self.stride,
                    self.se_ratio, self.nl_s, self.nl_c)



def get_active_fn(name):
    """Select activation function."""
    active_fn = {
        'nn.ReLU6': functools.partial(nn.ReLU6, inplace=True),
        'nn.ReLU': functools.partial(nn.ReLU, inplace=True),
        'nn.Swish': Swish,
        'nn.HSwish': HSwish,
    }[name]
    return active_fn


def get_nl_norm_fn(name):
    active_fn = {
        'nn.BatchNorm':
            ZeroInitBN,
        'nn.InstanceNorm':
            functools.partial(nn.InstanceNorm2d,
                              affine=True,
                              track_running_stats=True),
    }[name]
    return active_fn


def get_block(name):
    """Select building block."""
    return {
        # 'InvertedResidualChannels': InvertedResidualChannels,
        'InvertedResidualChannelsFused': InvertedResidualChannelsFused
    }[name]


def init_weights_slimmable(m):
    """Slimmable network style initialization."""
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        if m.affine:
            if isinstance(m, ZeroInitBN):
                nn.init.zeros_(m.weight)
            else:
                nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.zeros_(m.bias)


def init_weights_mnas(m):
    """MnasNet style initialization."""
    if isinstance(m, nn.Conv2d):
        if m.groups == m.in_channels:  # depthwise conv
            fan_out = m.weight[0][0].numel()
        else:
            _, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight)
        gain = nn.init.calculate_gain('relu')
        std = gain / math.sqrt(fan_out)
        nn.init.normal_(m.weight, 0.0, std)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        if m.affine:
            if isinstance(m, ZeroInitBN):
                nn.init.zeros_(m.weight)
            else:
                nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.InstanceNorm2d):
        if m.affine:
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        _, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight)
        init_range = 1.0 / np.sqrt(fan_out)
        nn.init.uniform_(m.weight, -init_range, init_range)
        nn.init.zeros_(m.bias)