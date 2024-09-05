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


class FBNetV2_6x_dnl(nn.Module):
    def __init__(
            self,
            nl_c=1.0,
            last_stride=2,
            batch_norm_momentum=0.1,
            batch_norm_epsilon=1e-3,
            nl_norm_method='batch_norm',
            **kwargs
    ):
        super(FBNetV2_6x_dnl, self).__init__()
        batch_norm_kwargs = {
            'momentum': batch_norm_momentum,
            'eps': batch_norm_epsilon
        }

        ori_model = fbnet("da_fbnetv2_6x", pretrained=False)
        fbnetv2_base = ori_model.backbone

        # construct layers
        block_num_stage = [5, 24, 42, 24]
        stage_end_index = [5, 5+24, 5+24+42, 5+24+42+24]

        self.stage_1 = fbnetv2_base.stages[0: stage_end_index[0]]
        self.out_channel_1 = fbnetv2_base.stages[stage_end_index[0] - 1].pwl.conv.out_channels
        self.non_local_1 = DNonlocal(self.out_channel_1, nl_c, 1, norm_method=nl_norm_method, batch_norm_kwargs=batch_norm_kwargs)

        self.stage_2 = fbnetv2_base.stages[stage_end_index[0]: stage_end_index[1]]
        self.out_channel_2 = fbnetv2_base.stages[stage_end_index[1] - 1].pwl.conv.out_channels
        self.non_local_2 = DNonlocal(self.out_channel_2, nl_c, 1, norm_method=nl_norm_method, batch_norm_kwargs=batch_norm_kwargs)

        self.stage_3 = fbnetv2_base.stages[stage_end_index[1]: stage_end_index[2]]
        self.out_channel_3 = fbnetv2_base.stages[stage_end_index[2] - 1].pwl.conv.out_channels
        self.non_local_3 = DNonlocal(self.out_channel_3, nl_c, 1, norm_method=nl_norm_method, batch_norm_kwargs=batch_norm_kwargs)

        self.stage_4 = fbnetv2_base.stages[stage_end_index[2]: stage_end_index[3]]
        if last_stride == 1:
            self.stage_4[0].dw.conv.stride = 1
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

    def featuremaps(self, x):
        x = self.stage_1(x)
        x = self.non_local_1(x)
        x = self.stage_2(x)
        x = self.non_local_2(x)
        x = self.stage_3(x)
        x = self.non_local_3(x)
        x = self.stage_4(x)
        x = self.non_local_4(x)
        x = self.stage_5(x)
        return x

    def forward(self, x):
        f = self.featuremaps(x)
        return f

def FBNetV2_6x_dnl_searched(last_stride=2, pre_train=False, **kwargs):
    model = FBNetV2_6x_dnl(last_stride=last_stride)

    searched_start_position = [10.163, 1.217, 32.331, 67.824]
    searched_length = [11.360, 30.915, 84.197, 127.325]

    model._init_params()

    for i in range(1, 5):
        model._set_ch_startpos_int(np.floor(searched_start_position[i-1]), i)
        model._set_ch_startpos_dec(searched_start_position[i-1] - np.floor(searched_start_position[i-1]), i)
    for i in range(1, 5):
        model._set_ch_length_int(np.floor(searched_length[i-1]), i)
        model._set_ch_length_dec(searched_length[i-1] - np.floor(searched_length[i-1]), i)

    for key, value in model.named_parameters():

        if 'ch_startpos_dec' in key:
            value.requires_grad = False
        if 'ch_length_dec' in key:
            value.requires_grad = False

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
