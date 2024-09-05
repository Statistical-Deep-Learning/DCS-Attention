#!/usr/bin/env python

import argparse
import pathlib
import time

import apex
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torchvision

from fvcore.common.checkpoint import Checkpointer

from pytorch_image_classification import (
    apply_data_parallel_wrapper,
    create_dataloader,
    create_loss,
    create_model,
    create_optimizer,
    create_scheduler,
    get_default_config,
    update_config,
)
from pytorch_image_classification.config.config_node import ConfigNode
from pytorch_image_classification.utils import (
    AverageMeter,
    DummyWriter,
    compute_accuracy,
    count_op,
    create_logger,
    create_tensorboard_writer,
    find_config_diff,
    get_env_info,
    get_rank,
    save_config,
    set_seed,
    setup_cudnn,
)

def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    config = get_default_config()
    if args.config is not None:
        config.merge_from_file(args.config)
    config.merge_from_list(args.options)
    if not torch.cuda.is_available():
        config.device = 'cpu'
        config.train.dataloader.pin_memory = False
    if args.resume != '':
        config_path = pathlib.Path(args.resume) / 'config.yaml'
        config.merge_from_file(config_path.as_posix())
        config.merge_from_list(['train.resume', True])
    config.merge_from_list(['train.dist.local_rank', args.local_rank])
    config = update_config(config)
    config.freeze()
    return config

def main():
    global global_step

    config = load_config()
    set_seed(config)
    setup_cudnn(config)
    model = create_model(config)
    old_one = torch.load('/home/ywan1053/reid-strong-baseline-master/imageNet_pretrained_models/checkpoint_00200.pth')
    model.load_state_dict(old_one['model'])
    torch.save(model.state_dict(), '/home/ywan1053/reid-strong-baseline-master/imageNet_pretrained_models/mobilenetv2-200.pth', _use_new_zipfile_serialization=False)
if __name__ == '__main__':
    main()