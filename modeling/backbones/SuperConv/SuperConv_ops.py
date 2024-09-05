import torch.nn as nn
import torch
import numpy as np
from torch.nn import functional as F
import warnings


class SuperConv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False, super_ratio=2, position_init=0.0):

        if groups == 1:
            super_group = 1
        else:
            super_group = int(groups * super_ratio)

        if in_channels == 3:
            super_in_channels = 3
        elif in_channels == 1:
            super_in_channels = 1
        else:
            super_in_channels = int(super_ratio * in_channels)

        super(SuperConv, self).__init__(
            super_in_channels, int(super_ratio * out_channels),
            kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=super_group, bias=bias)

        self.in_channels = in_channels
        self.width_mult = None
        self.super_ratio = super_ratio

        self.theta_1 = torch.nn.Parameter(torch.Tensor([1.0, 0.0, 0.0]), requires_grad=False)
        self.theta_2_0 = torch.nn.Parameter(torch.Tensor([0.0, 1.0]), requires_grad=False)

        self.position = torch.nn.Parameter(torch.Tensor([position_init]), requires_grad=True)

        self.pad_zero = torch.nn.Parameter(torch.Tensor([0.0]), requires_grad=False)
        self.groups = groups

    def forward(self, input):

        # Get the Super Filter Summary
        super_weight_shape = self.weight.shape
        super_length = super_weight_shape[0] * super_weight_shape[1] * super_weight_shape[2] * super_weight_shape[3]

        weight = self.weight.view(1, 1, super_length, 1)
        weight_replicate = weight
        weight_padded = torch.cat((weight, weight_replicate), dim=2)

        pad_zero = self.pad_zero.to(torch.float32).view(1, 1, 1, 1)

        weight_padded = torch.cat((weight_padded, pad_zero), dim=2)

        # Calculate the Transformation Theta Matrix
        cal_position = torch.sigmoid(self.position)

        size = torch.Size((1, 1, int(2 * super_length + 1), 1))

        theta_1 = self.theta_1
        theta_2 = torch.cat((self.theta_2_0, cal_position), dim=0)

        theta = torch.cat((theta_1, theta_2), dim=0)
        theta = theta.view(1, 2, 3)

        # Generate the Flow Field
        flow_field = torch.nn.functional.affine_grid(theta, size)
        sampled_weight = torch.nn.functional.grid_sample(weight_padded.to(torch.float32), flow_field.to(torch.float32))

        if self.groups == 1 and self.in_channels != 3 and self.in_channels != 1:
            compress_rate = self.super_ratio * self.super_ratio
            interpolated_weight = sampled_weight[:, :, 0:int(super_length / compress_rate), :].to(torch.float32)
            interpolated_weight = interpolated_weight.view(int(super_weight_shape[0] / 2),
                                                           int(super_weight_shape[1] / 2),
                                                           super_weight_shape[2],
                                                           super_weight_shape[3])

        else:
            compress_rate = self.super_ratio
            interpolated_weight = sampled_weight[:, :, 0:int(super_length / compress_rate), :].to(torch.float32)
            interpolated_weight = interpolated_weight.view(int(super_weight_shape[0] / 2), super_weight_shape[1],
                                                           super_weight_shape[2], super_weight_shape[3])

        # interpolated_weight = sampled_weight[:, :, 0:int(super_length/compress_rate), :].to(torch.float32)
        # interpolated_weight = interpolated_weight.view(int(super_weight_shape[0]/2), super_weight_shape[1], super_weight_shape[2], super_weight_shape[3])

        # Bias not implemented yet
        if self.bias is not None:
            bias = self.bias[:self.out_channels]
        else:
            bias = self.bias


        # Do convolution with sampled weights
        y = F.conv2d(
            input, interpolated_weight, bias, self.stride, self.padding,
            self.dilation, self.groups)
        # if getattr(FLAGS, 'conv_averaged', False):
        #     y = y * (max(self.in_channels_list) / self.in_channels)
        return y

    def print_position(self):
        return self.position


