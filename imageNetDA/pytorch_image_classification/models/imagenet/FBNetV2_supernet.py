from __future__ import division, absolute_import
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
from mobile_cv.model_zoo.models.fbnet_v2 import fbnet

class irf_stage(nn.Module):
    def __init__(self, supernet_blocks, option_number, option_step, largest_out_channel, block_number, out_H,
                               out_W, exp_rate=6, use_gpu=True):
        super(irf_stage, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(block_number):
            self.layers.append(block_unit(supernet_blocks[i], option_number, option_step, largest_out_channel, out_H,
                               out_W, exp_rate, use_gpu))

    def forward(self, x, temperature, effective_channel, cost_accumulate):
        for i in range(len(self.layers)):
            x, effective_channel, cost_accumulate = self.layers[i](x, temperature, effective_channel, cost_accumulate)
        return x, effective_channel, cost_accumulate


class block_unit(nn.Module):
    def __init__(
            self,
            irf_block_from_supernet,
            option_number,
            option_step,
            largest_out_channel,
            out_H,
            out_W,
            expansion_rate=6,
            use_gpu=True,
            **kwargs
    ):
        super(block_unit, self).__init__()
        self.block = irf_block_from_supernet
        self.GS_thetas = nn.Parameter(torch.Tensor([1.0 / option_number for i in range(option_number)]))
        self.option_step = option_step
        self.option_number = option_number
        self.largest_out_channel = largest_out_channel
        self.out_H = out_H
        self.out_W = out_W
        self.expansion_rate = expansion_rate
        self.use_gpu = use_gpu

    def masking(self, x, temperature):
        N, channel_in, H, W = list(x.shape)
        soft_mask_variables = nn.functional.gumbel_softmax(self.GS_thetas, temperature)
        mask = torch.zeros(N, self.largest_out_channel, self.out_H, self.out_W)
        mask = Variable(mask, requires_grad=False)
        if self.use_gpu:
            mask = mask.cuda()


        effective_output_channel = 0

        for i in range(self.option_number):
            if i == 0:
                mask_i = torch.ones(N, self.largest_out_channel, self.out_H, self.out_W)

            else:
                mask_i_one = torch.ones(N, self.largest_out_channel - i * self.option_step, self.out_H, self.out_W)
                mask_i_zero = torch.zeros(N, i * self.option_step, self.out_H, self.out_W)
                mask_i = torch.cat((mask_i_one, mask_i_zero), dim=1)

            mask_i = Variable(mask_i, requires_grad=False)
            if self.use_gpu:
                mask_i = mask_i.cuda()
            mask = mask + soft_mask_variables[i] * mask_i

            weighted_out_channel_i = soft_mask_variables[i] * (self.largest_out_channel - i * self.option_step)
            effective_output_channel = effective_output_channel + weighted_out_channel_i

        x = x.mul(mask)

        return x, effective_output_channel

    def calculate_irf_block_cost(self, effective_input_channel, effective_output_channel):

        pw_cost = self.out_H * self.out_W * effective_input_channel * effective_output_channel * self.expansion_rate

        dw_kernel_size = self.block.dw.conv.kernel_size[0]
        dw_cost = self.out_H * self.out_W * dw_kernel_size * dw_kernel_size * effective_output_channel * \
                  self.expansion_rate * effective_output_channel * self.expansion_rate

        pwl_cost = self.out_H * self.out_W * effective_output_channel * effective_output_channel * self.expansion_rate
        cost = pw_cost + dw_cost + pwl_cost

        return cost

    def forward(self, x, temperature, effective_input_channel, cost_before):

        x = self.block(x)
        x, effective_output_channel = self.masking(x, temperature)
        cost = self.calculate_irf_block_cost(effective_input_channel, effective_output_channel)
        cost_after = cost + cost_before

        return x, effective_output_channel, cost_after


class Network(nn.Module):
    def __init__(self, config):
    # def __init__(self):
        super(Network, self).__init__()

        supernet = fbnet("fbnetv2_supernet", pretrained=False)
        fbnetv2_base = supernet.backbone.stages

        option_numbers = [2, 4, 4, 7, 8, 14]
        option_steps = [4, 4, 8, 8, 8, 8]
        block_numbers = [1, 3, 3, 3, 4, 4]
        largest_out_channels = [16, 28, 40, 96, 128, 216]
        block_start_index = [1, 2, 5, 8, 11, 15]
        self.feature_size = config.model.feature_size
        self.out_H = config.dataset.image_size
        self.out_W = config.dataset.image_size
        self.out_H = 224
        self.out_W = 224
        if config.device != 'cpu':
            self.use_gpu = True
        else:
            self.use_gpu = False

        self.out_H = self.out_H // 2
        self.out_W = self.out_W // 2

        self.first_conv = fbnetv2_base[0]

        id_stage = 0

        self.irf_stage_1 = irf_stage(
            fbnetv2_base[block_start_index[id_stage]: block_start_index[id_stage] + block_numbers[id_stage]],
            option_numbers[id_stage],
            option_steps[id_stage],
            largest_out_channels[id_stage],
            block_numbers[id_stage],
            self.out_H,
            self.out_W,
            1,
            use_gpu=self.use_gpu
        )

        self.out_H = self.out_H // 2
        self.out_W = self.out_W // 2

        id_stage = 1

        self.irf_stage_2 = irf_stage(
            fbnetv2_base[block_start_index[id_stage]: block_start_index[id_stage] + block_numbers[id_stage]],
            option_numbers[id_stage],
            option_steps[id_stage],
            largest_out_channels[id_stage],
            block_numbers[id_stage],
            self.out_H,
            self.out_W,
            use_gpu=self.use_gpu
        )

        self.out_H = self.out_H // 2
        self.out_W = self.out_W // 2

        id_stage = 2

        self.irf_stage_3 = irf_stage(
            fbnetv2_base[block_start_index[id_stage]: block_start_index[id_stage] + block_numbers[id_stage]],
            option_numbers[id_stage],
            option_steps[id_stage],
            largest_out_channels[id_stage],
            block_numbers[id_stage],
            self.out_H,
            self.out_W,
            use_gpu=self.use_gpu
        )

        self.out_H = self.out_H // 2
        self.out_W = self.out_W // 2

        id_stage = 3

        self.irf_stage_4 = irf_stage(
            fbnetv2_base[block_start_index[id_stage]: block_start_index[id_stage] + block_numbers[id_stage]],
            option_numbers[id_stage],
            option_steps[id_stage],
            largest_out_channels[id_stage],
            block_numbers[id_stage],
            self.out_H,
            self.out_W,
            use_gpu=self.use_gpu
        )

        id_stage = 4

        self.irf_stage_5 = irf_stage(
            fbnetv2_base[block_start_index[id_stage]: block_start_index[id_stage] + block_numbers[id_stage]],
            option_numbers[id_stage],
            option_steps[id_stage],
            largest_out_channels[id_stage],
            block_numbers[id_stage],
            self.out_H,
            self.out_W,
            use_gpu=self.use_gpu
        )

        self.out_H = self.out_H // 2
        self.out_W = self.out_W // 2

        id_stage = 5

        self.irf_stage_6 = irf_stage(
            fbnetv2_base[block_start_index[id_stage]: block_start_index[id_stage] + block_numbers[id_stage]],
            option_numbers[id_stage],
            option_steps[id_stage],
            largest_out_channels[id_stage],
            block_numbers[id_stage],
            self.out_H,
            self.out_W,
            use_gpu=self.use_gpu
        )

        self.last_conv = fbnetv2_base[block_start_index[id_stage] + block_numbers[id_stage]]
        self.fc = nn.Linear(self.feature_size, config.dataset.n_classes)

        self._init_params()


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

    def _forward_conv(self, x, temperature, cost_accumulate):
        x = self.first_conv(x)
        effective_channel = 16
        x, effective_channel, cost_accumulate = self.irf_stage_1(x, temperature, effective_channel, cost_accumulate)
        x, effective_channel, cost_accumulate = self.irf_stage_2(x, temperature, effective_channel, cost_accumulate)
        x, effective_channel, cost_accumulate = self.irf_stage_3(x, temperature, effective_channel, cost_accumulate)
        x, effective_channel, cost_accumulate = self.irf_stage_4(x, temperature, effective_channel, cost_accumulate)
        x, effective_channel, cost_accumulate = self.irf_stage_5(x, temperature, effective_channel, cost_accumulate)
        x, effective_channel, cost_accumulate = self.irf_stage_6(x, temperature, effective_channel, cost_accumulate)
        x = self.last_conv(x)
        x = F.adaptive_avg_pool2d(x, output_size=1)
        return x, cost_accumulate

    def forward(self, x, temperature):

        cost_accumulate = Variable(torch.Tensor([[0.0]]), requires_grad=True)
        if self.use_gpu:
            cost_accumulate = cost_accumulate.cuda()
        x, cost_accumulate = self._forward_conv(x, temperature, cost_accumulate)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x, cost_accumulate
