from torch.nn import functional as F
import torch.nn as nn
from .NL_base import Nonlocal
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, k, s=1, p=0, g=1):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(
            in_c, out_c, k, stride=s, padding=p, bias=False, groups=g
        )

        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return F.relu6(self.bn(self.conv(x)))


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


class Network(nn.Module):

    def __init__(self, config):

        super().__init__()
        # self.loss = loss
        width_mult = 1.0
        last_stride = 2
        structure = [1, 2, 3, 4, 3, 3, 1]
        batch_norm_momentum = 0.1
        batch_norm_epsilon = 1e-3
        nl_c = 0.125
        nl_norm_method = 'batch_norm'
        batch_norm_kwargs = {
            'momentum': batch_norm_momentum,
            'eps': batch_norm_epsilon
        }
        self.in_channels = int(32 * width_mult)
        # self.feature_dim = int(1280 * width_mult) if width_mult > 1 else 1280
        self.feature_dim = int(1280 * width_mult)
        # construct layers
        self.conv1 = ConvBlock(3, self.in_channels, 3, s=2, p=1)
        self.conv2 = self._make_layer(Bottleneck, 1, int(16 * width_mult), structure[0], 1)
        self.conv3 = self._make_layer(Bottleneck, 6, int(24 * width_mult), structure[1], 2)
        self.non_local_1 = Nonlocal(int(24 * width_mult), nl_c, 1, norm_method=nl_norm_method,
                                    batch_norm_kwargs=batch_norm_kwargs)

        self.conv4 = self._make_layer(Bottleneck, 6, int(32 * width_mult), structure[2], 2)
        self.non_local_2 = Nonlocal(int(32 * width_mult), nl_c, 1, norm_method=nl_norm_method,
                                    batch_norm_kwargs=batch_norm_kwargs)

        self.conv5 = self._make_layer(Bottleneck, 6, int(64 * width_mult), structure[3], 2)
        self.non_local_3 = Nonlocal(int(64 * width_mult), nl_c, 1, norm_method=nl_norm_method,
                                    batch_norm_kwargs=batch_norm_kwargs)

        self.conv6 = self._make_layer(Bottleneck, 6, int(96 * width_mult), structure[4], 1)
        self.conv7 = self._make_layer(Bottleneck, 6, int(160 * width_mult), structure[5], last_stride)
        self.conv8 = self._make_layer(Bottleneck, 6, int(320 * width_mult), structure[6], 1)
        self.non_local_4 = Nonlocal(int(320 * width_mult), nl_c, 1, norm_method=nl_norm_method,
                                    batch_norm_kwargs=batch_norm_kwargs)

        self.conv9 = ConvBlock(self.in_channels, self.feature_dim, 1)
        self.fc = nn.Linear(self.feature_dim, config.dataset.n_classes)
        # self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        # self.fc = self._construct_fc_layer(fc_dims, self.feature_dim, dropout_p)
        # self.classifier = nn.Linear(self.feature_dim, num_classes)
        self._init_params()

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

    def featuremaps(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.non_local_1(x)

        x = self.conv4(x)
        x = self.non_local_2(x)

        x = self.conv5(x)
        x = self.non_local_3(x)

        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.non_local_4(x)

        x = self.conv9(x)
        return x

    def forward(self, x):
        x = self.featuremaps(x)
        x = F.adaptive_avg_pool2d(x, output_size=1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
