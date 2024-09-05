from __future__ import division, absolute_import
from torch.nn import functional as F
import torch.nn as nn
import warnings
import pickle
from functools import partial
import torch
import os.path as osp
from collections import OrderedDict


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

    def __init__(self, in_c, out_c, k, s=1, p=0, g=1, nonlinear_opt=True):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(in_c, out_c, k, stride=s, padding=p, bias=False, groups=g)
        self.bn = nn.BatchNorm2d(out_c)
        self.nonlinear_opt = nonlinear_opt

    def forward(self, x):
        if self.nonlinear_opt:
            return F.relu6(self.bn(self.conv(x)))
        else:
            return self.bn(self.conv(x))


class Bottleneck(nn.Module):

    def __init__(self, in_channels, out_channels, expansion_factor, stride=1):

        super(Bottleneck, self).__init__()

        mid_channels = in_channels * expansion_factor

        self.use_residual = stride == 1 and in_channels == out_channels

        self.conv1 = ConvBlock(in_channels, mid_channels, 1)

        self.dwconv2 = ConvBlock(mid_channels, mid_channels, 3, stride, 1, g=mid_channels)

        self.conv3 = ConvBlock(mid_channels, out_channels, 1, nonlinear_opt=False)

    def forward(self, x):

        m = self.conv1(x)

        m = self.dwconv2(m)

        m = self.conv3(m)

        if self.use_residual:
            return x + m

        else:
            return m


class Stage_gumbel(nn.Module):

    def __init__(self, expand_factor, out_channels, num_blocks, first_stride, channel_options):
        super(Stage_gumbel, self).__init__()

        self.layers = []
        self.layers.append(Bottleneck_gumbel(self.in_channels, out_channels, expand_factor,
                                        channel_options, first_stride))

        self.gumbel_weights = nn.Parameter(
            torch.Tensor([1.0 / len(channel_options) for i in range(len(channel_options))])
        )
        self.channel_options = channel_options
        self.in_channels = out_channels

        for i in range(1, num_blocks):
            self.layers.append(Bottleneck_gumbel(self.in_channels, out_channels, expand_factor, channel_options))

    def forward(self, x):

        for layer in self.layers:
            x = layer(x, self.gumbel_weights)

        expected_outch = 0

        for i in range(0, len(self.channel_options)):
            expected_outch = expected_outch + self.gumbel_weights[i] * self.channel_options[i]

        return x, expected_outch

# class Stage(nn.Module):
#
#     def __init__(self, in_channels, out_channels, block, expansion_rate, num_blocks, first_stride):
#         super(Stage, self).__init__()
#
#         layers = []
#         layers.append(block(in_channels, out_channels, expansion_rate, first_stride))
#         for i in range(1, num_blocks):
#             layers.append(block(in_channels, out_channels, expansion_rate))
#         return nn.Sequential(*layers)


class Bottleneck_gumbel(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor,
                 channel_options, stride=1):

        super(Bottleneck_gumbel, self).__init__()

        mid_channels = in_channels * expansion_factor

        self.expansion_factor = expansion_factor

        self.channel_options = channel_options

        self.num_options = len(channel_options)

        self.use_residual = stride == 1 and in_channels == out_channels

        self.conv1 = ConvBlock(in_channels, mid_channels, 1)

        self.dwconv2 = ConvBlock(mid_channels, mid_channels, 3, stride, 1, g=mid_channels)

        self.conv3 = ConvBlock(mid_channels, out_channels, 1, nonlinear_opt=False)

    def forward(self, x, gumbel_weights):

        m = self.conv1(x)

        for i in range(0, self.num_options):

            [batch, channels, height, width] = m.shape

            size_zero = [batch, channels - self.channel_options[i] * self.expansion_factor, height, width]

            size_one = [batch, self.channel_options[i] * self.expansion_factor, height, width]

            each_mask_0 = torch.zeros(size_zero, requires_grad=False)
            each_mask_1 = torch.ones(size_one, requires_grad=False)
            each_mask = torch.cat((each_mask_1, each_mask_0), dim=1)

            if i == 0:
                sum_mask = each_mask * gumbel_weights[i]
            else:
                sum_mask = sum_mask + each_mask * gumbel_weights[i]

        m = m * sum_mask

        m = self.dwconv2(m)

        for i in range(0, self.num_options):

            [batch, channels, height, width] = m.shape
            size_zero = [batch, channels - self.channel_options[i] * self.expansion_factor, height, width]
            size_one = [batch, self.channel_options[i] * self.expansion_factor, height, width]
            each_mask_0 = torch.zeros(size_zero, requires_grad=False)
            each_mask_1 = torch.ones(size_one, requires_grad=False)
            each_mask = torch.cat((each_mask_1, each_mask_0), dim=1)
            if i == 0:
                sum_mask = each_mask * gumbel_weights[i]
            else:
                sum_mask = sum_mask + each_mask * gumbel_weights[i]

        m = m * sum_mask

        m = self.conv3(m)

        for i in range(0, self.num_options):
            [batch, channels, height, width] = m.shape
            size_zero = [batch, channels - self.channel_options[i], height, width]
            size_one = [batch, self.channel_options[i], height, width]
            each_mask_0 = torch.zeros(size_zero, requires_grad=False)
            each_mask_1 = torch.ones(size_one, requires_grad=False)
            each_mask = torch.cat((each_mask_1, each_mask_0), dim=1)
            if i == 0:
                sum_mask = each_mask * gumbel_weights[i]
            else:
                sum_mask = sum_mask + each_mask * gumbel_weights[i]

        m = m * sum_mask

        if self.use_residual:
            return x + m

        else:
            return m


class MobileNetV2_deep(nn.Module):
    """MobileNetV2.

    Reference:
        Sandler et al. MobileNetV2: Inverted Residuals and
        Linear Bottlenecks. CVPR 2018.

    Public keys:
        - ``mobilenetv2_x1_0``: MobileNetV2 x1.0.
        - ``mobilenetv2_x1_4``: MobileNetV2 x1.4.
    """
    """
#################### Original Version of MobileNetV2 ######################

    Suppose the size of input is 224 x 224 x 3

                Input Size          Output Size         # Bottleneck (17)   # Conv Layers (53)
        conv1:  224 x 224 x 3       112 x 112 x 32      0                   1
        conv2:  112 x 112 x 32      112 x 112 x 16      1                   3

Stage1  conv3:  112 x 112 x 16      56  x 56  x 24      2                   6

Stage2  conv4:  56  x 56  x 24      28  x 28  x 32      3                   9

Stage3  conv5:  28  x 28  x 32      14  x 14  x 64      4                   12
        conv6:  14  x 14  x 64      14  x 14  x 96      3                   9

Stage4  conv7:  14  x 14  x 96       7  x  7  x 160     3                   9
        conv8:   7  x  7  x 160      7  x  7  x 320     1                   3

        conv9:   7  x  7  x 320      7  x  7  x 1280    0                   1

#################### Deeper Version of MobileNetV2 ######################

    The number of bottleneck blocks is defined in structure[].
                                    # Bottleneck at each stage              # Bottleneck
                                                                                 __  __  ______  _____                 
    MobileNetV2-53(Original)        [2,  3,  7,  4]                         [1,  2,  3,  4,  3,  3,  1]                                                                                     
    MobileNetV2-107                 [3,  4, 23,  4]                         [1,  3,  4, 12, 11,  3,  1]     
    MobileNetV2-161                 [3,  8, 37,  4]                         [1,  3,  8, 19, 18,  3,  1]     
    MobileNetV2-200                 [3, 21, 37,  4]                         [1,  3, 21, 19, 18,  3,  1]
    MobileNetV2-299                 [3, 31, 60,  4]                         [1,  3, 31, 30, 30,  3,  1]

    ResNet-50                       [3,  4,  6,  3]
    ResNet-101                      [3,  4, 23,  3]
    ResNet-152                      [3,  8, 36,  3]
    ResNet-200                      [3, 24, 36,  3]

#################### MobileNetV2 with last_stride = 1 ######################

    Modify the stride of Stage-4 from 2 to 1

                Input Size          Output Size         # Bottleneck (17)   # Conv Layers (53)
        conv1:  224 x 224 x 3       112 x 112 x 32      0                   1
        conv2:  112 x 112 x 32      112 x 112 x 16      1                   3

Stage1  conv3:  112 x 112 x 16      56  x 56  x 24      2                   6

Stage2  conv4:  56  x 56  x 24      28  x 28  x 32      3                   9

Stage3  conv5:  28  x 28  x 32      14  x 14  x 64      4                   12
        conv6:  14  x 14  x 64      14  x 14  x 96      3                   9

Stage4  conv7:  14  x 14  x 96      14  x 14  x 160     3                   9
        conv8:  14  x 14  x 160     14  x 14  x 320     1                   3

        conv9:  14  x 14  x 320     14  x 14  x 1280    0                   1
    """

    def __init__(
            self,
            channel_options=[[16], [24], [32], [64], [96], [160], [320]],
            width_mult=1.0,
            structure=[1, 2, 3, 4, 3, 3, 1],
            last_stride=2,
            **kwargs
    ):
        super(MobileNetV2_deep, self).__init__()
        # self.loss = loss
        self.in_channels = int(32 * width_mult)
        # self.feature_dim = int(1280 * width_mult) if width_mult > 1 else 1280
        self.feature_dim = int(1280 * width_mult)
        # construct layers
        self.conv1 = ConvBlock(3, self.in_channels, 3, s=2, p=1)

        self.conv2 = Stage_gumbel(1, int(16 * width_mult), structure[0], 1, channel_options[0])
        # self.conv2 = self._make_layer(Bottleneck, 1, int(16 * width_mult), structure[0], 1)

        self.conv3 = Stage_gumbel(6, int(24 * width_mult), structure[1], 2, channel_options[1])
        # self.conv3 = self._make_layer(Bottleneck, 6, int(24 * width_mult), structure[1], 2)

        self.conv4 = Stage_gumbel(6, int(32 * width_mult), structure[2], 2, channel_options[2])
        # self.conv4 = self._make_layer(Bottleneck, 6, int(32 * width_mult), structure[2], 2)

        self.conv5 = Stage_gumbel(6, int(64 * width_mult), structure[3], 2, channel_options[3])
        # self.conv5 = self._make_layer(Bottleneck, 6, int(64 * width_mult), structure[3], 2)

        self.conv6 = Stage_gumbel(6, int(96 * width_mult), structure[4], 1, channel_options[4])
        # self.conv6 = self._make_layer(Bottleneck, 6, int(64 * width_mult), structure[3], 2)

        self.conv7 = Stage_gumbel(6, int(96 * width_mult), structure[5], last_stride, channel_options[5])
        # self.conv7 = self._make_layer(Bottleneck, 6, int(160 * width_mult), structure[5], last_stride)

        self.conv8 = Stage_gumbel(6, int(320 * width_mult), structure[6], 1, channel_options[6])
        # self.conv8 = self._make_layer(Bottleneck, 6, int(320 * width_mult), structure[6], 1)

        self.conv9 = ConvBlock(self.in_channels, self.feature_dim, 1)

        self._init_params()

    # def _make_layer(self, block, expand_factor, out_channels, num_blocks, first_stride):
    #     layers = []
    #     layers.append(block(self.in_channels, out_channels, expand_factor, first_stride))
    #     self.in_channels = out_channels
    #     for i in range(1, num_blocks):
    #         layers.append(block(self.in_channels, out_channels, expand_factor))
    #     return nn.Sequential(*layers)
    #
    # def _make_gumbel_layer(self, expand_factor, out_channels, num_blocks, first_stride,
    #                        channel_options, gumbel_weights):
    #     layers = []
    #     layers.append(Bottleneck_gumbel(self.in_channels, out_channels, expand_factor,
    #                                     channel_options, gumbel_weights, first_stride))
    #     self.in_channels = out_channels
    #     for i in range(1, num_blocks):
    #         layers.append(Bottleneck_gumbel(self.in_channels, out_channels, expand_factor,
    #                                         channel_options, gumbel_weights))
    #     return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x, expected_outc2 = self.conv2(x)
        x, expected_outc3 = self.conv3(x)
        x, expected_outc4 = self.conv4(x)
        x, expected_outc5 = self.conv5(x)
        x, expected_outc6 = self.conv6(x)
        x, expected_outc7 = self.conv7(x)
        x, expected_outc8 = self.conv8(x)
        x = self.conv9(x)
        return x


def mobilenetv2_53_CS(channel_options, last_stride=2, width_mult=1.0, **kwargs):
    model = MobileNetV2_deep(
        channel_options=channel_options,
        width_mult=width_mult,
        structure=[1, 2, 3, 4, 3, 3, 1],
        last_stride=last_stride,
        **kwargs
    )
    warnings.warn("Training mobilenetv2_53 from scratch.")
    return model


def compute_stage_gumbel_flops(expected_inc, expected_outc, expand_factor, num_blocks, first_stride, in_height, in_width):
    return 0


def compute_gumbel_bottleneck_flops(expected_inc, expected_outc, expand_factor, stride, in_height, in_width):
    expected_midc = expected_outc * expand_factor
    flops_1 = compute_conv_flops(in_height, in_width, expected_inc, expected_midc, stride, kernel_size=1, groups=1)


def compute_conv_flops(in_height, in_width, e_in_ch, e_out_ch, stride, kernel_size, groups,
                        padding_height=0, padding_width=0, bias_ops=0):
    kernel_height = kernel_size
    kernel_width = kernel_size
    in_channel = e_in_ch
    out_channel = e_out_ch
    stride_height = stride
    stride_width = stride
    kernel_ops = kernel_height * kernel_width * (in_channel / groups)
    output_height = (in_height + padding_height * 2 - kernel_height) // stride_height + 1
    output_width = (in_width + padding_width * 2 - kernel_width) // stride_width + 1
    flops = (kernel_ops + bias_ops) * output_height * output_width * out_channel
    return flops


