from __future__ import division, absolute_import
import numbers
import torch.nn as nn
from torch.nn import functional as F
from modeling.backbones.searched_mobilenetv2.mobilenet_base import _make_divisible
from modeling.backbones.searched_mobilenetv2.mobilenet_base import ConvBNReLU
from modeling.backbones.searched_mobilenetv2.mobilenet_base import get_active_fn
from modeling.backbones.searched_mobilenetv2.mobilenet_base import get_block
from modeling.backbones.searched_mobilenetv2.mobilenet_base import InvertedResidualChannelsFused


class MobileNetV2(nn.Module):
    """MobileNetV2-like network."""

    def __init__(self,
                 num_classes=1000,
                 # input_size=224,
                 input_channel=32,
                 last_channel=1280,
                 width_mult=1.0,
                 inverted_residual_setting=None,
                 dropout_ratio=0.2,
                 batch_norm_momentum=0.1,
                 batch_norm_epsilon=1e-3,
                 active_fn='nn.ReLU6',
                 block='InvertedResidualChannels',
                 round_nearest=8,
                 **kwargs

                 ):
        """Build the network.
        Args:
            num_classes (int): Number of classes
            input_size (int): Input resolution.
            input_channel (int): Number of channels for stem convolution.
            last_channel (int): Number of channels for stem convolution.
            width_mult (float): Width multiplier - adjusts number of channels in
                each layer by this amount
            inverted_residual_setting (list): A list of
                [expand ratio, output channel, num repeat,
                stride of first block, A list of kernel sizes].
            dropout_ratio (float): Dropout ratio for linear classifier.
            batch_norm_momentum (float): Momentum for batch normalization.
            batch_norm_epsilon (float): Epsilon for batch normalization.
            active_fn (str): Specify which activation function to use.
            block (str): Specify which MobilenetV2 block implementation to use.
            round_nearest (int): Round the number of channels in each layer to
                be a multiple of this number Set to 1 to turn off rounding.
        """
        super(MobileNetV2, self).__init__()

        batch_norm_kwargs = {
            'momentum': batch_norm_momentum,
            'eps': batch_norm_epsilon
        }

        self.input_channel = input_channel
        self.last_channel = last_channel
        self.width_mult = width_mult
        self.round_nearest = round_nearest
        self.inverted_residual_setting = inverted_residual_setting
        self.active_fn = active_fn
        self.block = block

        # if len(inverted_residual_setting) == 0 or (len(
        #         inverted_residual_setting[0]) not in [5, 8]):
        #     raise ValueError("inverted_residual_setting should be non-empty "
        #                      "or a 5/8-element list, got {}".format(inverted_residual_setting))
        # if input_size % 32 != 0:
        #     raise ValueError('Input size must divide 32')

        # block = get_block_wrapper(block)

        # building first layer

        active_fn = get_active_fn(active_fn)

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult,
                                        round_nearest)
        last_channel = _make_divisible(last_channel * max(1.0, width_mult),
                                       round_nearest)
        features = [
            ConvBNReLU(3,
                       input_channel,
                       stride=2,
                       batch_norm_kwargs=batch_norm_kwargs,
                       active_fn=active_fn)
        ]
        # building inverted residual blocks
        for t, c, n, s, ks, nl_c, nl_s, se_ratio in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(
                    InvertedResidualChannelsFused(
                        inp=input_channel,
                        oup=output_channel,
                        stride=stride,
                        channels=[int(t*input_channel)],
                        kernel_sizes=ks,
                        expand=True,
                        active_fn=active_fn,
                        batch_norm_kwargs=batch_norm_kwargs,
                        se_ratio=se_ratio,
                        nl_c=nl_c,
                        nl_s=nl_s,
                    )
                )
                input_channel = output_channel
        # building last several layers
        features.append(
            ConvBNReLU(input_channel,
                       last_channel,
                       kernel_size=1,
                       batch_norm_kwargs=batch_norm_kwargs,
                       active_fn=active_fn))

        # avg_pool_size = input_size // 32
        # features.append(nn.AvgPool2d(avg_pool_size))
        # features.append(nn.AdaptiveAvgPool2d(1))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.bottleneck = nn.BatchNorm1d(self.last_channel)
        self.bottleneck.bias.requires_grad_(False)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_ratio),
            nn.Linear(last_channel, num_classes),
        )

    # def forward(self, x):
    #     x = self.features(x)
    #
    #     x = x.squeeze(3).squeeze(2)
    #     x = self.classifier(x)
    #     return x
    def forward(self, x):

        x = self.features(x)


        global_feat = self.gap(x)
        global_feat = global_feat.view(global_feat.shape[0], -1)

        if not self.training:
            return global_feat

        feat = self.bottleneck(global_feat)
        cls_score = self.classifier(feat)
        return cls_score, global_feat

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





def autonl_l(num_classes=751, last_stride=1, **kwargs):
    active_fn = 'nn.Swish'
    input_channel = 32
    last_channel = 1280
    width_mult = 1.0
    round_nearest = 8
    block = 'InvertedResidualChannelsFused'
    inverted_residual_setting = [
        [1, 16, 1, 1, [3], 0, 1, 0],
        [3, 24, 1, 2, [5], 0.25, 1, 0],
        [3, 24, 1, 1, [5], 0, 1, 0],
        [3, 24, 1, 1, [3], 0.25, 2, 0],
        [6, 40, 1, 2, [5], 0.25, 1, 0],
        [3, 40, 1, 1, [3], 0.25, 2, 0],
        [3, 40, 1, 1, [3], 0.25, 2, 0],
        [3, 40, 1, 1, [3], 0.25, 2, 0],
        [6, 80, 1, 2, [5], 0.25, 1, 0],
        [3, 80, 1, 1, [3], 0., 1, 0.5],
        [3, 80, 1, 1, [3], 0., 1, 0.5],
        [3, 80, 1, 1, [3], 0.125, 2, 0.5],
        [6, 96, 1, 1, [5], 0.25, 1, 0.5],
        [3, 96, 1, 1, [5], 0., 1, 0.5],
        [3, 96, 1, 1, [5], 0., 1, 0.5],
        [3, 96, 1, 1, [3], 0.25, 2, 0.5],
        [6, 192, 1, last_stride, [5], 0.25, 1, 0.5],
        [6, 192, 1, 1, [5], 0.125, 1, 0.5],
        [6, 192, 1, 1, [5], 0.125, 1, 0.5],
        [6, 192, 1, 1, [5], 0, 1, 0.5],
        [6, 320, 1, 1, [5], 0.25, 1, 0.5]
    ]

    model = MobileNetV2(num_classes=num_classes,
                        active_fn=active_fn,
                        input_channel=input_channel,
                        last_channel=last_channel,
                        width_mult=width_mult,
                        round_nearest=round_nearest,
                        block=block,
                        inverted_residual_setting=inverted_residual_setting,
                        **kwargs
                        )

    model._init_params()

    return model


def mobilenetv2_nl(num_classes=751, last_stride=1, nl_c=0.25, **kwargs):
    active_fn = 'nn.ReLU'
    input_channel = 32
    last_channel = 1280
    width_mult = 1.0
    round_nearest = 8
    block = 'InvertedResidualChannelsFused'
    ls = last_stride


    inverted_residual_setting = [
        [1, 16, 1, 1, [3], nl_c, 2, 0],
        [6, 24, 1, 2, [3], nl_c, 2, 0],
        [6, 24, 1, 1, [3], nl_c, 2, 0],
        [6, 32, 1, 2, [3], nl_c, 2, 0],
        [6, 32, 1, 1, [3], nl_c, 2, 0],
        [6, 32, 1, 1, [3], nl_c, 2, 0],
        [6, 64, 1, 2, [3], nl_c, 1, 0],
        [6, 64, 1, 1, [3], nl_c, 1, 0],
        [6, 64, 1, 1, [3], nl_c, 1, 0],
        [6, 64, 1, 1, [3], nl_c, 1, 0],
        [6, 96, 1, 1, [3], nl_c, 1, 0],
        [6, 96, 1, 1, [3], nl_c, 1, 0],
        [6, 96, 1, 1, [3], nl_c, 1, 0],
        [6, 160, 1, ls, [3], nl_c, 1, 0],
        [6, 160, 1, 1, [3], nl_c, 1, 0],
        [6, 160, 1, 1, [3], nl_c, 1, 0],
        [6, 320, 1, 1, [3], nl_c, 1, 0],
    ]

    model = MobileNetV2(num_classes=num_classes,
                        active_fn=active_fn,
                        input_channel=input_channel,
                        last_channel=last_channel,
                        width_mult=width_mult,
                        round_nearest=round_nearest,
                        block=block,
                        inverted_residual_setting=inverted_residual_setting,
                        **kwargs
                        )

    model._init_params()

    return model

def mobilenetv2_baseline(num_classes=751, last_stride=1, **kwargs):
    active_fn = 'nn.ReLU'
    input_channel = 32
    last_channel = 1280
    width_mult = 1.0
    round_nearest = 8
    block = 'InvertedResidualChannelsFused'
    ls = last_stride
    inverted_residual_setting = [
        [1, 16, 1, 1, [3], 0, 0, 0],
        [6, 24, 1, 2, [3], 0, 0, 0],
        [6, 24, 1, 1, [3], 0, 0, 0],
        [6, 32, 1, 2, [3], 0, 0, 0],
        [6, 32, 1, 1, [3], 0, 0, 0],
        [6, 32, 1, 1, [3], 0, 0, 0],
        [6, 64, 1, 2, [3], 0, 0, 0],
        [6, 64, 1, 1, [3], 0, 0, 0],
        [6, 64, 1, 1, [3], 0, 0, 0],
        [6, 64, 1, 1, [3], 0, 0, 0],
        [6, 96, 1, 1, [3], 0, 0, 0],
        [6, 96, 1, 1, [3], 0, 0, 0],
        [6, 96, 1, 1, [3], 0, 0, 0],
        [6, 160, 1, ls, [3], 0, 0, 0],
        [6, 160, 1, 1, [3], 0, 0, 0],
        [6, 160, 1, 1, [3], 0, 0, 0],
        [6, 320, 1, 1, [3], 0, 0, 0]
    ]

    model = MobileNetV2(num_classes=num_classes,
                        active_fn=active_fn,
                        input_channel=input_channel,
                        last_channel=last_channel,
                        width_mult=width_mult,
                        round_nearest=round_nearest,
                        block=block,
                        inverted_residual_setting=inverted_residual_setting,
                        **kwargs
                        )

    model._init_params()

    return model

def mobilenetv2_stage_nl(num_classes=751, last_stride=1, **kwargs):
    active_fn = 'nn.ReLU'
    input_channel = 32
    last_channel = 1280
    width_mult = 1.0
    round_nearest = 8
    block = 'InvertedResidualChannelsFused'
    ls = last_stride
    nl_c = 0.25
    inverted_residual_setting = [
        [1, 16, 1, 1, [3], 0, 0, 0],
        [6, 24, 1, 2, [3], 0, 0, 0],
        [6, 24, 1, 1, [3], nl_c, 2, 0],
        [6, 32, 1, 2, [3], 0, 0, 0],
        [6, 32, 1, 1, [3], 0, 0, 0],
        [6, 32, 1, 1, [3], 0, 0, 0],
        [6, 64, 1, 2, [3], nl_c, 2, 0],
        [6, 64, 1, 1, [3], 0, 0, 0],
        [6, 64, 1, 1, [3], 0, 0, 0],
        [6, 64, 1, 1, [3], 0, 0, 0],
        [6, 96, 1, 1, [3], nl_c, 1, 0],
        [6, 96, 1, 1, [3], 0, 0, 0],
        [6, 96, 1, 1, [3], 0, 0, 0],
        [6, 160, 1, ls, [3], nl_c, 1, 0],
        [6, 160, 1, 1, [3], 0, 0, 0],
        [6, 160, 1, 1, [3], 0, 0, 0],
        [6, 320, 1, 1, [3], 0, 0, 0]
    ]

    model = MobileNetV2(num_classes=num_classes,
                        active_fn=active_fn,
                        input_channel=input_channel,
                        last_channel=last_channel,
                        width_mult=width_mult,
                        round_nearest=round_nearest,
                        block=block,
                        inverted_residual_setting=inverted_residual_setting,
                        **kwargs
                        )

    model._init_params()

    return model