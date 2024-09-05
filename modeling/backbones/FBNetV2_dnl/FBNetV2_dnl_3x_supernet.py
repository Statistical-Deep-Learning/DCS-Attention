from __future__ import division, absolute_import
from torch.nn import functional as F
import torch.nn as nn
import warnings
import pickle
from .DNLblock import DNonlocal
from functools import partial
import torch
import os.path as osp
from collections import OrderedDict
import numpy as np
from mobile_cv.model_zoo.models.fbnet_v2 import fbnet

class blocks_unit(nn.Module):
    def __init__(
            self,
            blocks_from_supernet,
            option_number,
            block_number,
            option_steps,
            out_channel,
            out_H,
            out_W,
            expansion_rate=6,
            **kwargs
    ):
        super(blocks_unit, self).__init__()
        self.blocks = blocks_from_supernet
        self.GS_thetas = nn.Parameter(torch.Tensor([1.0 / option_number for i in range(option_number)]))
        self.block_number = block_number
        self.option_steps = option_steps
        self.option_number = option_number
        self.out_channel = out_channel
        self.out_H = out_H
        self.out_W = out_W
        self.expansion_rate = expansion_rate


    def masking(self, x, temperature):
        N, channel_in, H, W = list(x.shape)
        soft_mask_variables = nn.functional.gumbel_softmax(self.GS_thetas, temperature)
        mask = torch.zeros(N, self.out_channel, self.out_H, self.out_W)
        effective_output_channel = 0
        for i in range(self.option_number):
            if i == 0:
                mask_i = torch.ones(N, self.out_channel, self.out_H, self.out_W)

            else:
                mask_i_one = torch.ones(N, self.out_channel - i * self.option_steps, self.out_H, self.out_W)
                mask_i_zero = torch.zeros(N, i * self.option_steps, self.out_H, self.out_W)
                mask_i = torch.cat((mask_i_one, mask_i_zero), dim=1)
            mask = mask + soft_mask_variables[i] * mask_i

            weighted_out_channel_i = soft_mask_variables[i] * (self.out_channel - i * self.option_steps)
            effective_output_channel = effective_output_channel + weighted_out_channel_i

        x = x.mul(mask)
        return x, effective_output_channel

    def forward(self, x, temperature):

        if self.block_number == 1:
            x = self.blocks(x)
            x, effective_output_channel = self.masking(x, temperature)
        else:
            for block in self.blocks:
                x = block(x)
                x, effective_output_channel = self.masking(x, temperature)

        pw_cost = effective_output_channel * self.expansion_rate
        dw_cost = 3 * 3 * effective_output_channel * self.expansion_rate * effective_output_channel * self.expansion_rate
        pwl_cost = effective_output_channel * self.expansion_rate
        cost = pw_cost + dw_cost + pwl_cost
        return x, cost

class FBNetV2_3x_dnl_supernet(nn.Module):
    def __init__(
            self,
            nl_c=1.0,
            last_stride=1,
            batch_norm_momentum=0.1,
            batch_norm_epsilon=1e-3,
            nl_norm_method='batch_norm',
            **kwargs
    ):
        super(FBNetV2_3x_dnl_supernet, self).__init__()
        batch_norm_kwargs = {
            'momentum': batch_norm_momentum,
            'eps': batch_norm_epsilon
        }

        ori_model = fbnet("da_fbnetv2_3x_supernet", pretrained=False)
        fbnetv2_base = ori_model.backbone

        # construct layers
        block_num_stage = [5, 12, 21, 12]
        stage_end_index = [5, 5+12, 5+12+21, 5+12+21+12]

        out_H = 384
        out_W = 128
        out_H = out_H // 2
        out_W = out_W // 2

        self.conv1 = fbnetv2_base.stages[0: 2]

        out_H = out_H // 2
        out_W = out_W // 2

        self.blocks_unit_1 = blocks_unit(fbnetv2_base.stages[2: 5],
                                         option_number=4,
                                         block_number=3,
                                         option_steps=4,
                                         out_channel=28,
                                         out_H=out_H,
                                         out_W=out_W
                                         )

        self.out_channel_1 = fbnetv2_base.stages[stage_end_index[0] - 1].pwl.conv.out_channels
        self.non_local_1 = DNonlocal(self.out_channel_1, nl_c, 1, norm_method=nl_norm_method, batch_norm_kwargs=batch_norm_kwargs)

        out_H = out_H // 2
        out_W = out_W // 2

        self.blocks_unit_2 = blocks_unit(fbnetv2_base.stages[stage_end_index[0]: stage_end_index[1]],
                                         option_number=4,
                                         block_number=stage_end_index[1] - stage_end_index[0],
                                         option_steps=8,
                                         out_channel=40,
                                         out_H=out_H,
                                         out_W=out_W
                                         )

        self.out_channel_2 = fbnetv2_base.stages[stage_end_index[1] - 1].pwl.conv.out_channels
        self.non_local_2 = DNonlocal(self.out_channel_2, nl_c, 1, norm_method=nl_norm_method, batch_norm_kwargs=batch_norm_kwargs)

        out_H = out_H // 2
        out_W = out_W // 2

        self.blocks_unit_3 = blocks_unit(fbnetv2_base.stages[stage_end_index[1]: stage_end_index[2]],
                                         option_number=7,
                                         block_number=stage_end_index[2] - stage_end_index[1],
                                         option_steps=8,
                                         out_channel=128,
                                         out_H=out_H,
                                         out_W=out_W
                                         )

        self.out_channel_3 = fbnetv2_base.stages[stage_end_index[2] - 1].pwl.conv.out_channels
        self.non_local_3 = DNonlocal(self.out_channel_3, nl_c, 1, norm_method=nl_norm_method, batch_norm_kwargs=batch_norm_kwargs)

        if last_stride == 2:
            out_H = out_H // 2
            out_W = out_W // 2
        self.blocks_unit_4 = blocks_unit(fbnetv2_base.stages[stage_end_index[2]: stage_end_index[3]],
                                         option_number=13,
                                         block_number=stage_end_index[3] - stage_end_index[2],
                                         option_steps=8,
                                         out_channel=216,
                                         out_H=out_H,
                                         out_W=out_W
                                         )
        if last_stride == 1:
            self.blocks_unit_4.blocks[0].dw.conv.stride = 1

        self.out_channel_4 = fbnetv2_base.stages[stage_end_index[3] - 1].pwl.conv.out_channels
        self.non_local_4 = DNonlocal(self.out_channel_4, nl_c, 1, norm_method=nl_norm_method, batch_norm_kwargs=batch_norm_kwargs)

        self.stage_5 = fbnetv2_base.stages[stage_end_index[3]]
        self._init_params()

    def _set_ch_startpos_int(self, ch_startpos_int, index):
        if torch.cuda.is_available():
            startpos_int = torch.from_numpy(np.array(ch_startpos_int)).cuda().to(torch.int64)
        else:
            startpos_int = torch.from_numpy(np.array(ch_startpos_int)).to(torch.int64)
        layer_eval = eval('self.non_local_' + str(index) + '.ch_startpos_int')
        layer_eval.data = startpos_int

    def _set_ch_startpos_dec(self, ch_startpos_dec, index):
        if torch.cuda.is_available():
            startpos_dec = torch.from_numpy(np.array(ch_startpos_dec)).cuda().float()
        else:
            startpos_dec = torch.from_numpy(np.array(ch_startpos_dec)).float()
        layer_eval = eval('self.non_local_' + str(index) + '.ch_startpos_dec')
        layer_eval.data = startpos_dec

    def _set_ch_length_int(self, ch_length_int, index):
        if torch.cuda.is_available():
            length_int = torch.from_numpy(np.array(ch_length_int)).cuda().to(torch.int64)
        else:
            length_int = torch.from_numpy(np.array(ch_length_int)).to(torch.int64)
        layer_eval = eval('self.non_local_' + str(index) + '.ch_length_int')
        layer_eval.data = length_int

    def _set_ch_length_dec(self, ch_length_dec, index):
        if torch.cuda.is_available():
            length_dec = torch.from_numpy(np.array(ch_length_dec)).cuda().float()
        else:
            length_dec = torch.from_numpy(np.array(ch_length_dec)).float()
        layer_eval = eval('self.non_local_' + str(index) + '.ch_length_dec')
        layer_eval.data = length_dec

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


    def forward(self, x, temperature):
        x = self.conv1(x)
        x, cost1 = self.blocks_unit_1(x, temperature)
        x = self.non_local_1(x)
        x, cost2 = self.blocks_unit_2(x, temperature)
        x = self.non_local_2(x)
        x, cost3 = self.blocks_unit_3(x, temperature)
        x = self.non_local_3(x)
        x, cost4 = self.blocks_unit_4(x, temperature)
        x = self.non_local_4(x)
        x = self.stage_5(x)
        cost = cost1 + cost2 + cost3 + cost4
        return x, cost
