# encoding: utf-8

import argparse
import os
import sys
from functools import partial
from os import mkdir

import torch
from torch.backends import cudnn

sys.path.append('.')
from config import cfg
from data import make_data_loader
from engine.inference import inference
from modeling import build_model
import warnings
from utils.logger import setup_logger
import os.path as osp
import pickle
from collections import OrderedDict



def main():
    parser = argparse.ArgumentParser(description="ReID Baseline Inference")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        mkdir(output_dir)

    logger = setup_logger("reid_baseline", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DEVICE == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    cudnn.benchmark = True

    train_loader, val_loader, num_query, num_classes = make_data_loader(cfg)
    model = build_model(cfg, num_classes)
    model.load_param(cfg.TEST.WEIGHT)
    # load_pretrained_weights(model, cfg.TEST.WEIGHT)

    inference(cfg, model, val_loader, num_query)

# def load_pretrained_weights(model, weight_path):
#     r"""Loads pretrianed weights to model.
#     Features::
#         - Incompatible layers (unmatched in name or size) will be ignored.
#         - Can automatically deal with keys containing "module.".
#     Args:
#         model (nn.Module): network model.
#         weight_path (str): path to pretrained weights.
#     Examples::
#        #>>> from torchreid.utils import load_pretrained_weights
#        #>>> weight_path = 'log/my_model/model-best.pth.tar'
#        #>>> load_pretrained_weights(model, weight_path)
#     """
#     checkpoint = load_checkpoint(weight_path)
#     if 'state_dict' in checkpoint:
#         state_dict = checkpoint['state_dict']
#     else:
#         state_dict = checkpoint
#
#     model_dict = model.state_dict()
#
#     new_state_dict = OrderedDict()
#     matched_layers, discarded_layers = [], []
#
#     # for k, v in state_dict.items():
#     #     warnings.warn("The size of " + k + " is " + v.size())
#
#     for k, v in state_dict.items():
#         if k.startswith('module.'):
#             k = k[7:]  # discard module.
#
#         if k in model_dict and model_dict[k].size() == v.size():
#             new_state_dict[k] = v
#             matched_layers.append(k)
#         else:
#             discarded_layers.append(k)
#
#     model_dict.update(new_state_dict)
#     model.load_state_dict(model_dict)
#
#     if len(matched_layers) == 0:
#         warnings.warn(
#             'The pretrained weights "{}" cannot be loaded, '
#             'please check the key names manually '
#             '(** ignored and continue **)'.format(weight_path)
#         )
#     else:
#         print(
#             'Successfully loaded pretrained weights from "{}"'.
#                 format(weight_path)
#         )
#         if len(discarded_layers) > 0:
#             print(
#                 '** The following layers are discarded '
#                 'due to unmatched keys or layer size: {}'.
#                     format(discarded_layers)
#             )
#
#
# def load_checkpoint(fpath):
#     r"""Loads checkpoint.
#     ``UnicodeDecodeError`` can be well handled, which means
#     python2-saved files can be read from python3.
#     Args:
#         fpath (str): path to checkpoint.
#     Returns:
#         dict
#     Examples::
#         #>>> from torchreid.utils import load_checkpoint
#         #>>> fpath = 'log/my_model/model.pth.tar-10'
#         #>>> checkpoint = load_checkpoint(fpath)
#     """
#     if fpath is None:
#         raise ValueError('File path is None')
#     if not osp.exists(fpath):
#         raise FileNotFoundError('File is not found at "{}"'.format(fpath))
#     map_location = None if torch.cuda.is_available() else 'cpu'
#     try:
#         checkpoint = torch.load(fpath, map_location=map_location)
#     except UnicodeDecodeError:
#         pickle.load = partial(pickle.load, encoding="latin1")
#         pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
#         checkpoint = torch.load(
#             fpath, pickle_module=pickle, map_location=map_location
#         )
#     except Exception:
#         print('Unable to load checkpoint from "{}"'.format(fpath))
#         raise
#     return checkpoint

if __name__ == '__main__':
    main()

