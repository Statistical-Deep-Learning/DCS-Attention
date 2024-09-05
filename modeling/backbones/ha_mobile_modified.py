from __future__ import division, absolute_import
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F
import pickle
import os.path as osp
import warnings
from functools import partial
from collections import OrderedDict
import torch
import torch.nn as nn
import warnings

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)



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

class InceptionA(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(InceptionA, self).__init__()
        mid_channels = out_channels // 4

        self.stream1 = nn.Sequential(
            ConvBlock(in_channels, mid_channels, 1),
            ConvBlock(mid_channels, mid_channels, 3, p=1),
        )
        self.stream2 = nn.Sequential(
            ConvBlock(in_channels, mid_channels, 1),
            ConvBlock(mid_channels, mid_channels, 3, p=1),
        )
        self.stream3 = nn.Sequential(
            ConvBlock(in_channels, mid_channels, 1),
            ConvBlock(mid_channels, mid_channels, 3, p=1),
        )
        self.stream4 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            ConvBlock(in_channels, mid_channels, 1),
        )

    def forward(self, x):
        s1 = self.stream1(x)
        s2 = self.stream2(x)
        s3 = self.stream3(x)
        s4 = self.stream4(x)
        y = torch.cat([s1, s2, s3, s4], dim=1)
        return y


class InceptionB(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(InceptionB, self).__init__()
        mid_channels = out_channels // 4

        self.stream1 = nn.Sequential(
            ConvBlock(in_channels, mid_channels, 1),
            ConvBlock(mid_channels, mid_channels, 3, s=2, p=1),
        )
        self.stream2 = nn.Sequential(
            ConvBlock(in_channels, mid_channels, 1),
            ConvBlock(mid_channels, mid_channels, 3, p=1),
            ConvBlock(mid_channels, mid_channels, 3, s=2, p=1),
        )
        self.stream3 = nn.Sequential(
            nn.MaxPool2d(3, stride=2, padding=1),
            ConvBlock(in_channels, mid_channels * 2, 1),
        )

    def forward(self, x):
        s1 = self.stream1(x)
        s2 = self.stream2(x)
        s3 = self.stream3(x)
        y = torch.cat([s1, s2, s3], dim=1)
        return y



class SpatialAttn(nn.Module):
    """Spatial Attention (Sec. 3.1.I.1)"""

    def __init__(self):
        super(SpatialAttn, self).__init__()
        self.conv1 = ConvBlock(1, 1, 3, s=2, p=1)
        self.conv2 = ConvBlock(1, 1, 1)

    def forward(self, x):
        # global cross-channel averaging
        x = x.mean(1, keepdim=True)
        # 3-by-3 conv
        x = self.conv1(x)
        # bilinear resizing
        x = F.upsample(
            x, (x.size(2) * 2, x.size(3) * 2),
            mode='bilinear',
            align_corners=True
        )
        # scaling conv
        x = self.conv2(x)
        return x


class ChannelAttn(nn.Module):
    """Channel Attention (Sec. 3.1.I.2)"""

    def __init__(self, in_channels, reduction_rate=8):
        super(ChannelAttn, self).__init__()
        assert in_channels % reduction_rate == 0
        self.conv1 = ConvBlock(in_channels, in_channels // reduction_rate, 1)
        self.conv2 = ConvBlock(in_channels // reduction_rate, in_channels, 1)

    def forward(self, x):
        # squeeze operation (global average pooling)
        x = F.avg_pool2d(x, x.size()[2:])
        # excitation operation (2 conv layers)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SoftAttn(nn.Module):
    """Soft Attention (Sec. 3.1.I)

    Aim: Spatial Attention + Channel Attention

    Output: attention maps with shape identical to input.
    """

    def __init__(self, in_channels):
        super(SoftAttn, self).__init__()
        self.spatial_attn = SpatialAttn()
        self.channel_attn = ChannelAttn(in_channels)
        self.conv = ConvBlock(in_channels, in_channels, 1)

    def forward(self, x):
        y_spatial = self.spatial_attn(x)
        y_channel = self.channel_attn(x)
        y = y_spatial * y_channel
        y = torch.sigmoid(self.conv(y))
        return y


class HardAttn(nn.Module):
    """Hard Attention (Sec. 3.1.II)"""

    def __init__(self, in_channels):
        super(HardAttn, self).__init__()
        self.fc = nn.Linear(in_channels, 4 * 2)
        self.init_params()

    def init_params(self):
        self.fc.weight.data.zero_()
        self.fc.bias.data.copy_(
            torch.tensor(
                [0, -0.75, 0, -0.25, 0, 0.25, 0, 0.75], dtype=torch.float
            )
        )

    def forward(self, x):
        # squeeze operation (global average pooling)
        x = F.avg_pool2d(x, x.size()[2:]).view(x.size(0), x.size(1))
        # predict transformation parameters
        theta = torch.tanh(self.fc(x))
        theta = theta.view(-1, 4, 2)
        return theta


class HarmAttn(nn.Module):
    """Harmonious Attention (Sec. 3.1)"""

    def __init__(self, in_channels):
        super(HarmAttn, self).__init__()
        self.soft_attn = SoftAttn(in_channels)
        self.hard_attn = HardAttn(in_channels)

    def forward(self, x):
        y_soft_attn = self.soft_attn(x)
        theta = self.hard_attn(x)
        return y_soft_attn, theta


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


class HAMobile(nn.Module):
    """MobileNetV2.

    Reference:
        Sandler et al. MobileNetV2: Inverted Residuals and
        Linear Bottlenecks. CVPR 2018.

    Public keys:
        - ``mobilenetv2_x1_0``: MobileNetV2 x1.0.
        - ``mobilenetv2_x1_4``: MobileNetV2 x1.4.
    """

    def __init__(
            self,
            num_classes,
            local_resolution=[24, 28],
            width_mult=1,
            first_stride=2,
            last_stride=1,
            learn_region=True,
            use_gpu=True,
            global_attn=True,
            neck_feat='after',
            neck='bnneck',
            fc_dims=None,
            dropout_p=None,
            **kwargs
    ):
        super(HAMobile, self).__init__()
        #self.loss = loss
        feat_dim = 1280
        self.global_attn = global_attn
        self.local_resolution = local_resolution
        self.learn_region = learn_region
        self.use_gpu = use_gpu
        self.neck_feat = neck_feat
        self.neck = neck
        nchannels = [0, 0, 0, 0]

        self.in_channels = int(32 * width_mult)
        self.feature_dim = int(1280 * width_mult) if width_mult > 1 else 1280


        self.feature_dim = 1280



        # construct layers
        self.conv1 = ConvBlock(3, self.in_channels, 3, s=2, p=1)
        self.conv2 = self._make_layer(Bottleneck, 1, int(16 * width_mult), 1, 1)
        self.conv3 = self._make_layer(Bottleneck, 6, int(24 * width_mult), 2, first_stride)

        nchannels[0] = int(24 * width_mult)

        self.conv4 = self._make_layer(Bottleneck, 6, int(32 * width_mult), 3, 2)

        nchannels[1] = int(32 * width_mult)

        self.conv5 = self._make_layer(Bottleneck, 6, int(64 * width_mult), 4, 2)
        self.conv6 = self._make_layer(Bottleneck, 6, int(96 * width_mult), 3, 1)
        self.conv7 = self._make_layer(Bottleneck, 6, int(160 * width_mult), 3, last_stride)
        self.conv8 = self._make_layer(Bottleneck, 6, int(320 * width_mult), 1, 1)
        self.conv9 = ConvBlock(self.in_channels, self.feature_dim, 1)

        nchannels[2] = int(320 * width_mult)

        # self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        # self.fc = self._construct_fc_layer(fc_dims, self.feature_dim, dropout_p)

        self.fc_global = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.BatchNorm1d(self.feature_dim),
            nn.ReLU(),
        )


        self.classifier_global = nn.Linear(self.feature_dim, num_classes)
        # self.classifier = nn.Linear(self.feature_dim, num_classes)
        self._init_params()
        self.ha1 = HarmAttn(nchannels[0])
        self.ha2 = HarmAttn(nchannels[1])
        self.ha3 = HarmAttn(nchannels[2])

        if self.neck == 'bnneck':
            if self.learn_region:
                self.bottleneck_local = nn.BatchNorm1d(self.feature_dim)
                self.bottleneck_local.bias.requires_grad_(False)  # no shift
                self.bottleneck_local.apply(weights_init_kaiming)

            self.bottleneck_global = nn.BatchNorm1d(self.feature_dim)
            self.bottleneck_global.bias.requires_grad_(False)  # no shift
            self.bottleneck_global.apply(weights_init_kaiming)

        if self.learn_region:
            self.init_scale_factors()
            self.local_conv1 = InceptionB(32, nchannels[0])
            self.local_conv2 = InceptionB(nchannels[0], nchannels[1])
            self.local_conv3 = InceptionB(nchannels[1], nchannels[2])
            self.fc_local = nn.Sequential(
                nn.Linear(nchannels[2] * 4, self.feature_dim),
                nn.BatchNorm1d(self.feature_dim),
                nn.ReLU(),
            )
            self.classifier_local = nn.Linear(self.feature_dim, num_classes)
        #     self.feat_dim = feat_dim * 2
        # else:
        #     self.feat_dim = feat_dim


    def _make_layer_local(self, block, t, in_c, out_c, n, s):
        # t: expansion factor
        # c: output channels
        # n: number of blocks
        # s: stride for first layer
        layers = []
        layers.append(block(in_c, out_c, t, s))
        for i in range(1, n):
            layers.append(block(out_c, out_c, t))
        return nn.Sequential(*layers)

    def _make_layer(self, block, t, c, n, s):
        # t: expansion factor
        # c: output channels
        # n: number of blocks
        # s: stride for first layer
        layers = []
        layers.append(block(self.in_channels, c, t, s))
        self.in_channels = c
        for i in range(1, n):
            layers.append(block(self.in_channels, c, t))
        return nn.Sequential(*layers)

    def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
        """Constructs fully connected layer.

        Args:
            fc_dims (list or tuple): dimensions of fc layers, if None, no fc layers are constructed
            input_dim (int): input dimension
            dropout_p (float): dropout probability, if None, dropout is unused
        """
        if fc_dims is None:
            self.feature_dim = input_dim
            return None

        assert isinstance(
            fc_dims, (list, tuple)
        ), 'fc_dims must be either list or tuple, but got {}'.format(
            type(fc_dims)
        )

        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout_p is not None:
                layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim

        self.feature_dim = fc_dims[-1]

        return nn.Sequential(*layers)

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


    def init_scale_factors(self):
        # initialize scale factors (s_w, s_h) for four regions
        self.scale_factors = []
        self.scale_factors.append(
            torch.tensor([[1, 0], [0, 0.25]], dtype=torch.float)
        )
        self.scale_factors.append(
            torch.tensor([[1, 0], [0, 0.25]], dtype=torch.float)
        )
        self.scale_factors.append(
            torch.tensor([[1, 0], [0, 0.25]], dtype=torch.float)
        )
        self.scale_factors.append(
            torch.tensor([[1, 0], [0, 0.25]], dtype=torch.float)
        )

    def stn(self, x, theta):
        """Performs spatial transform

        x: (batch, channel, height, width)
        theta: (batch, 2, 3)
        """
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x

    def transform_theta(self, theta_i, region_idx):
        """Transforms theta to include (s_w, s_h), resulting in (batch, 2, 3)"""
        scale_factors = self.scale_factors[region_idx]
        theta = torch.zeros(theta_i.size(0), 2, 3)
        theta[:, :, :2] = scale_factors
        theta[:, :, -1] = theta_i
        if self.use_gpu: theta = theta.cuda()
        return theta

    def forward(self, x):

        x = self.conv1(x)

        # ------The First Downsample Block-------------------

        # global branch
        x1 = self.conv2(x)
        x1 = self.conv3(x1)
        x1_attn, x1_theta = self.ha1(x1)

        if self.global_attn:
            x1_out = x1 * x1_attn
        else:
            x1_out = x1

        local_height = self.local_resolution[0]
        local_width = self.local_resolution[1]

        # local branch
        if self.learn_region:
            x1_local_list = []
            for region_idx in range(4):
                x1_theta_i = x1_theta[:, region_idx, :]
                x1_theta_i = self.transform_theta(x1_theta_i, region_idx)
                x1_trans_i = self.stn(x, x1_theta_i)
                x1_trans_i = F.upsample(
                    x1_trans_i, (int(local_height), int(local_width)), mode='bilinear', align_corners=True
                )
                x1_local_i = self.local_conv1(x1_trans_i)
                x1_local_list.append(x1_local_i)

        # ------The Second Downsample Block-------------------

        # global branch
        x2 = self.conv4(x1_out)
        x2_attn, x2_theta = self.ha2(x2)

        if self.global_attn:
            x2_out = x2 * x2_attn
        else:
            x2_out = x2

        # local branch
        if self.learn_region:
            x2_local_list = []
            for region_idx in range(4):
                x2_theta_i = x2_theta[:, region_idx, :]
                x2_theta_i = self.transform_theta(x2_theta_i, region_idx)
                x2_trans_i = self.stn(x1_out, x2_theta_i)
                x2_trans_i = F.upsample(
                    x2_trans_i, (int(local_height/2), int(local_width/2)), mode='bilinear', align_corners=True
                )
                x2_local_i = x2_trans_i + x1_local_list[region_idx]
                x2_local_i = self.local_conv2(x2_local_i)
                x2_local_list.append(x2_local_i)

        # ------The Third Downsample Block-------------------

        # global branch
        x3 = self.conv5(x2_out)
        x3 = self.conv6(x3)
        x3 = self.conv7(x3)
        x3 = self.conv8(x3)

        x3_attn, x3_theta = self.ha3(x3)

        if self.global_attn:
            x3_out = x3 * x3_attn
        else:
            x3_out = x3

        x3_out = self.conv9(x3_out)

        # local branch
        if self.learn_region:
            x3_local_list = []
            for region_idx in range(4):
                x3_theta_i = x3_theta[:, region_idx, :]
                x3_theta_i = self.transform_theta(x3_theta_i, region_idx)
                x3_trans_i = self.stn(x2_out, x3_theta_i)
                x3_trans_i = F.upsample(
                    x3_trans_i, (int(local_height/4), int(local_width/4)), mode='bilinear', align_corners=True
                )
                x3_local_i = x3_trans_i + x2_local_list[region_idx]
                x3_local_i = self.local_conv3(x3_local_i)
                x3_local_list.append(x3_local_i)

        x_global = F.avg_pool2d(x3_out, x3_out.size()[2:]).view(x3_out.size(0), x3_out.size(1))
        x_global = self.fc_global(x_global)

        if self.learn_region:
            x_local_list = []

            for region_idx in range(4):
                x_local_i = x3_local_list[region_idx]
                x_local_i = F.avg_pool2d(x_local_i, x_local_i.size()[2:]).view(x_local_i.size(0), -1)
                x_local_list.append(x_local_i)

            x_local = torch.cat(x_local_list, 1)
            x_local = self.fc_local(x_local)

        # go through BN neck
        if self.neck == 'no':
            global_feat_2cls = x_global
            if self.learn_region:
                local_feat_2cls = x_local
        elif self.neck == 'bnneck':
            global_feat_2cls = self.bottleneck_global(x_global)  # normalize for angular softmax
            if self.learn_region:
                local_feat_2cls = self.bottleneck_local(x_local)

        # test phase
        if not self.training:
            if self.neck_feat == 'after':
                if self.learn_region:
                    global_feat_2cls = global_feat_2cls / global_feat_2cls.norm(p=2, dim=1, keepdim=True)
                    local_feat_2cls = local_feat_2cls / local_feat_2cls.norm(p=2, dim=1, keepdim=True)
                    return torch.cat([global_feat_2cls, local_feat_2cls], 1)
                else:
                    return global_feat_2cls
            else:
                if self.learn_region:
                    x_global = x_global / x_global.norm(p=2, dim=1, keepdim=True)
                    x_local = x_local / x_local.norm(p=2, dim=1, keepdim=True)
                    return torch.cat([x_global, x_local], 1)
                else:
                    return x_global

        # training phase
        prelogits_global = self.classifier_global(global_feat_2cls)
        if self.learn_region:
            prelogits_local = self.classifier_local(local_feat_2cls)
            return prelogits_global, x_global, prelogits_local, x_local
        else:
            return prelogits_global, x_global


def ha_mobilenet_modified_global_local(cfg, num_classes, use_gpu=True, **kwargs):
    model = HAMobile(
        num_classes,
        local_resolution=[40, 56],
        width_mult=1,
        first_stride=2,
        last_stride=1,
        neck_feat=cfg.TEST.NECK_FEAT,
        neck=cfg.MODEL.NECK,
        learn_region=True,
        use_gpu=use_gpu,
        **kwargs
    )
    return model


def ha_mobilenet_modified_global_only(cfg, num_classes, use_gpu=True, **kwargs):
    model = HAMobile(
        num_classes,
        local_resolution=[40, 56],
        width_mult=1,
        first_stride=2,
        last_stride=1,
        neck_feat=cfg.TEST.NECK_FEAT,
        neck=cfg.MODEL.NECK,
        learn_region=False,
        use_gpu=use_gpu,
        **kwargs
    )
    return model

def ha_mobilenet_modified_local(cfg, num_classes, use_gpu=True, **kwargs):
    model = HAMobile(
        num_classes,
        local_resolution=[40, 56],
        width_mult=1,
        first_stride=2,
        last_stride=1,
        neck_feat=cfg.TEST.NECK_FEAT,
        neck=cfg.MODEL.NECK,
        global_attn=False,
        learn_region=True,
        use_gpu=use_gpu,
        **kwargs
    )
    return model
