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
            width_mult=1,
            #loss='softmax',
            learn_region=True ,
            use_gpu=True,
            fc_dims=None,
            dropout_p=None,
            **kwargs
    ):
        super(HAMobile, self).__init__()
        #self.loss = loss
        feat_dim = 1280
        self.learn_region = learn_region
        self.use_gpu= use_gpu
        nchannels = [0, 0, 0, 0]
        self.in_channels = int(32 * width_mult)
        self.feature_dim = int(1280 * width_mult) if width_mult > 1 else 1280

        # construct layers
        self.conv1 = ConvBlock(3, self.in_channels, 3, s=2, p=1)
        self.conv2 = self._make_layer(Bottleneck, 1, int(16 * width_mult), 1, 1)
        self.conv3 = self._make_layer(Bottleneck, 6, int(24 * width_mult), 2, 2)
        nchannels[0] = int(24 * width_mult)
        self.conv4 = self._make_layer(Bottleneck, 6, int(32 * width_mult), 3, 2)
        nchannels[1] = int(32 * width_mult)
        self.conv5 = self._make_layer(Bottleneck, 6, int(64 * width_mult), 4, 2)
        nchannels[2] = int(64 * width_mult)
        self.conv6 = self._make_layer(Bottleneck, 6, int(96 * width_mult), 3, 1)
        self.conv7 = self._make_layer(Bottleneck, 6, int(160 * width_mult), 3, 2)

        self.conv8 = self._make_layer(Bottleneck, 6, int(320 * width_mult), 1, 1)
        self.conv9 = ConvBlock(self.in_channels, self.feature_dim, 1)

        # self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        # self.fc = self._construct_fc_layer(fc_dims, self.feature_dim, dropout_p)

        self.fc_global = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.BatchNorm1d(self.feature_dim),
            nn.ReLU(),
        )
        # self.classifier = nn.Linear(self.feature_dim, num_classes)
        self._init_params()
        self.ha1 = HarmAttn(nchannels[0])
        self.ha2 = HarmAttn(nchannels[1])
        self.ha3 = HarmAttn(nchannels[2])
        nchannels[3] = 1280

        if self.learn_region:
            self.init_scale_factors()
            self.local_conv1 = InceptionB(32, nchannels[0])
            self.local_conv2 = InceptionB(nchannels[0], nchannels[1])
            self.local_conv3 = InceptionB(nchannels[1], nchannels[2])
            self.local_conv4 = InceptionB(nchannels[2], nchannels[3])
            self.fc_local = nn.Sequential(
                nn.Linear(nchannels[2] * 4, self.feature_dim),
                nn.BatchNorm1d(self.feature_dim),
                nn.ReLU(),
            )

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

    # def featuremaps(self, x):
    #     x = self.conv1(x)
    #     x = self.conv2(x)
    #     x = self.conv3(x)
    #     x = self.conv4(x)
    #     x = self.conv5(x)
    #     x = self.conv6(x)
    #     x = self.conv7(x)
    #     x = self.conv8(x)
    #     x = self.conv9(x)
    #     return x
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
        x1_out = x1 * x1_attn

        local_height = 40
        local_width = 56

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
        x2_out = x2 * x2_attn

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
        x3_attn, x3_theta = self.ha3(x3)
        x3_out = x3 * x3_attn

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

        # ------The Fourth Block---------------------------------

        x4 = self.conv6(x3_out)
        x4 = self.conv7(x4)
        x4 = self.conv8(x4)
        x4_out = self.conv9(x4)

        # if self.learn_region:
        #     x4_local_list = []
        #     for region_idx in range(4):
        #         x4_local_i = self.local_conv4(x3_local_list[region_idx])
        #         x4_local_list.append(x4_local_i)
        #

        x_global = F.avg_pool2d(x4_out, x4_out.size()[2:]).view(x4_out.size(0), x4_out.size(1))
        x_global = self.fc_global(x_global)

        if self.learn_region:
            x_local_list = []
            for region_idx in range(4):
                x_local_i = x3_local_list[region_idx]
                x_local_i = F.avg_pool2d(x_local_i, x_local_i.size()[2:]).view(x_local_i.size(0), -1)
                x_local_list.append(x_local_i)
            x_local = torch.cat(x_local_list, 1)
            x_local = self.fc_local(x_local)

        if self.learn_region:
            x_global = x_global / x_global.norm(p=2, dim=1, keepdim=True)
            x_local = x_local / x_local.norm(p=2, dim=1, keepdim=True)
            return torch.cat([x_global, x_local], 1)
        else:
            return x_global

        # v = self.global_avgpool(f)
        # v = v.view(v.size(0), -1)
        #
        # if self.fc is not None:
        #     v = self.fc(v)
        #
        # if not self.training:
        #     return v

        # y = self.classifier(v)
        #
        # if self.loss == 'softmax':
        #     return y
        # elif self.loss == 'triplet':
        #     return y, v
        # else:
        #     raise KeyError("Unsupported loss: {}".format(self.loss))


def hamobilenet(num_classes, pretrain_choice='imagenet', **kwargs):
    model = HAMobile(
        num_classes,
        # loss=loss,
        width_mult=1,
        **kwargs
    )

    if pretrain_choice == 'imagenet':
        load_pretrained_weights(model, 'imageNet_pretrained_models/mobilenetv2_1dot0.pth.tar')
        warnings.warn('Pretained model loaded.')

    return model




def load_pretrained_weights(model, weight_path):
    r"""Loads pretrianed weights to model.
    Features::
        - Incompatible layers (unmatched in name or size) will be ignored.
        - Can automatically deal with keys containing "module.".
    Args:
        model (nn.Module): network model.
        weight_path (str): path to pretrained weights.
    Examples::
       #>>> from torchreid.utils import load_pretrained_weights
       #>>> weight_path = 'log/my_model/model-best.pth.tar'
       #>>> load_pretrained_weights(model, weight_path)
    """
    checkpoint = load_checkpoint(weight_path)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    model_dict = model.state_dict()

    new_state_dict = OrderedDict()
    matched_layers, discarded_layers = [], []

    # for k, v in state_dict.items():
    #     warnings.warn("The size of " + k + " is " + v.size())

    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]  # discard module.

        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)

    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)

    if len(matched_layers) == 0:
        warnings.warn(
            'The pretrained weights "{}" cannot be loaded, '
            'please check the key names manually '
            '(** ignored and continue **)'.format(weight_path)
        )
    else:
        print(
            'Successfully loaded pretrained weights from "{}"'.
                format(weight_path)
        )
        if len(discarded_layers) > 0:
            print(
                '** The following layers are discarded '
                'due to unmatched keys or layer size: {}'.
                    format(discarded_layers)
            )


def load_checkpoint(fpath):
    r"""Loads checkpoint.
    ``UnicodeDecodeError`` can be well handled, which means
    python2-saved files can be read from python3.
    Args:
        fpath (str): path to checkpoint.
    Returns:
        dict
    Examples::
        #>>> from torchreid.utils import load_checkpoint
        #>>> fpath = 'log/my_model/model.pth.tar-10'
        #>>> checkpoint = load_checkpoint(fpath)
    """
    if fpath is None:
        raise ValueError('File path is None')
    if not osp.exists(fpath):
        raise FileNotFoundError('File is not found at "{}"'.format(fpath))
    map_location = None if torch.cuda.is_available() else 'cpu'
    try:
        checkpoint = torch.load(fpath, map_location=map_location)
    except UnicodeDecodeError:
        pickle.load = partial(pickle.load, encoding="latin1")
        pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
        checkpoint = torch.load(
            fpath, pickle_module=pickle, map_location=map_location
        )
    except Exception:
        print('Unable to load checkpoint from "{}"'.format(fpath))
        raise
    return checkpoint





