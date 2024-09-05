# encoding: utf-8


import torch


def make_optimizer(cfg, model):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.MOMENTUM)
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params)
    return optimizer


def make_optimizer_with_center(cfg, model, center_criterion):

    params = []

    for key, value in model.named_parameters():

        # Only optimize parameters that require gradients
        if not value.requires_grad:
            continue

        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY

        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS

        if "position" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.POSITION_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_POSITION

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.MOMENTUM)
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params)

    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=cfg.SOLVER.CENTER_LR)
    return optimizer, optimizer_center



def make_dnl_optimizer_with_center(cfg, model, center_criterion):

    params = []
    dnl_params = []
    for key, value in model.named_parameters():

        # Only optimize parameters that require gradients
        if not value.requires_grad:
            continue

        if "ch_startpos_dec" in key or "ch_length_dec" in key:
            dnl_params += [{"params": [value], "lr": cfg.DNL.STARTPOS_LR, "weight_decay": cfg.DNL.STARTPOS_WEIGHT_DECAY}]
            continue
        #
        # if "ch_length_dec" in key:
        #     dnl_params += [{"params": [value], "lr": cfg.DNL.LENGTH_LR, "weight_decay": cfg.DNL.LENGTH_WEIGHT_DECAY}]
        #     continue

        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY

        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS

        if "position" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.POSITION_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_POSITION

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.MOMENTUM)
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params)

    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=cfg.SOLVER.CENTER_LR)

    if cfg.MODEL.DNL:
        if cfg.DNL.OPTIMIZER_TYPE == 'SGD':
            optimizer_dnl = getattr(torch.optim, cfg.DNL.OPTIMIZER_TYPE)(dnl_params, momentum=cfg.SOLVER.MOMENTUM)
        else:
            optimizer_dnl = getattr(torch.optim, cfg.DNL.OPTIMIZER_TYPE)(dnl_params)

        return optimizer, optimizer_center, optimizer_dnl
    else:
        return optimizer, optimizer_center
