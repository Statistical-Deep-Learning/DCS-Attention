import torch.nn as nn
import torch
import numpy as np
from torch.nn import functional as F
from torch.autograd import Variable
import scipy.special
import warnings


class multiW_SuperConv(nn.Conv2d):
    def __init__(self, super_in_channels, super_out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False,
                 num_window=4, window_length=16, real_inchannels=0, real_outchannels=0):

        if real_inchannels == 0:
            real_inchannels = window_length
        if real_outchannels == 0:
            real_outchannels = num_window

        if groups == 1:
            super_group = 1
        else:
            super_group = super_in_channels

        if groups == 1:
            self.real_groups = 1
        else:
            self.real_groups = real_inchannels

        super(multiW_SuperConv, self).__init__(
            super_in_channels, super_out_channels,
            kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=super_group, bias=bias)

        self.real_inchannels = real_inchannels
        self.real_outchannels = real_outchannels
        self.num_window = num_window
        self.window_length = window_length
        self.pad_zero = torch.nn.Parameter(torch.Tensor([0.0]), requires_grad=False)

        super_weight_shape = self.weight.shape
        super_length = super_weight_shape[0] * super_weight_shape[1] * super_weight_shape[2] * super_weight_shape[3]

        position_np = np.array([])

        for i in range(0, num_window):
            # bound_index_sum = 2 * ((i + 1) * window_length - 1) - (window_length - 1)
            bound_index_sum = 2 * ((i + 1) * int(super_length / self.num_window) - 1) - (int(super_length / self.num_window) - 1)
            related_cal_position = (bound_index_sum / (super_length - 1)) - 1
            related_position = (related_cal_position + 1) / 2
            related_position = scipy.special.logit(related_position)
            position_np = np.append(position_np, related_position)

        position_array = torch.from_numpy(position_np)
        self.position = torch.nn.Parameter(position_array.to(torch.float32), requires_grad=True)
        self.theta_0 = torch.nn.Parameter(torch.Tensor([1.0, 0.0, 0.0, 0.0, (window_length - 1) / (super_length - 1)]),
                                          requires_grad=False)

    def forward(self, input):

        # Get the Super Filter Summary
        super_weight_shape = self.weight.shape
        super_length = super_weight_shape[0] * super_weight_shape[1] * super_weight_shape[2] * super_weight_shape[3]
        weight = self.weight.view(1, 1, super_length, 1)

        # Calculate the Transformation Theta Matrix
        cal_position = 2 * torch.sigmoid(self.position) - 1
        cal_position_chunk = torch.chunk(cal_position, self.num_window)

        size = torch.Size((1, 1, self.window_length, 1))
        theta_0 = self.theta_0

        for i in range(0, self.num_window):

            theta = torch.cat((theta_0, cal_position_chunk[i]), dim=0)
            theta = theta.view(1, 2, 3)

            if i == 0:
                flow_field = torch.nn.functional.affine_grid(theta, size)
            else:
                flow_field = torch.cat((flow_field, torch.nn.functional.affine_grid(theta, size)), dim=1)

        interpolated_weight_all = torch.nn.functional.grid_sample(weight.to(torch.float32),
                                                                  flow_field.to(torch.float32))
        if self.real_groups == 1:
            interpolated_weight_all = interpolated_weight_all.view(self.real_outchannels,
                                                                   self.real_inchannels,
                                                                   super_weight_shape[2],
                                                                   super_weight_shape[3])
        else:
            interpolated_weight_all = interpolated_weight_all.view(self.real_outchannels,
                                                                   int(self.real_inchannels/self.real_groups),
                                                                   super_weight_shape[2],
                                                                   super_weight_shape[3])

        # Bias not implemented yet (MobileNetV2 does not use bias.)
        if self.bias is not None:
            bias = self.bias[:self.out_channels]
        else:
            bias = self.bias

        # Do convolution with sampled weights
        y = F.conv2d(
            input, interpolated_weight_all, bias, self.stride, self.padding,
            self.dilation, self.real_groups)

        return y



