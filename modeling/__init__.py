# encoding: utf-8


from .baseline import Baseline, Search_Base
import torch

def build_model(cfg, num_classes):

    if cfg.SOLVER.SEARCH_FBNETV2:
        model = Search_Base(num_classes, cfg)
    else:
        model = Baseline(num_classes, cfg)

    return model
