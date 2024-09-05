# encoding: utf-8
import argparse
import os
import sys
import torch
from torch.backends import cudnn
sys.path.append('.')
from config import cfg
from data import make_data_loader
from engine.searcher import do_search
from modeling import build_model
from layers import make_loss, make_loss_with_center
from solver import make_optimizer, make_optimizer_with_center, WarmupMultiStepLR, \
    make_dnl_optimizer_with_center, WarmupMultiStepLR_reinit, CosineAnnealingLR
from utils import model_complexity
from utils.logger import setup_logger


def load_param(pretrain, trained_path):
    param_dict = torch.load(trained_path)
    for k, v in param_dict.state_dict().items():
        pretrain.state_dict()[k].copy_(param_dict.state_dict()[k])
    return pretrain

def train_supernet(cfg):
    # prepare dataset
    train_loader, val_loader, num_query, num_classes = make_data_loader(cfg)

    # prepare model
    model = build_model(cfg, num_classes)

    if cfg.MODEL.COMPUTE_MODEL_COMPLEXITY:
        size_train = cfg.INPUT.SIZE_TRAIN
        model_complexity.compute_model_complexity(
            model,
            (1, 3, size_train[0], size_train[1]),
            verbose=True,
            only_conv_linear=False
        )


    loss_func, center_criterion = make_loss_with_center(cfg, num_classes)
    optimizer, optimizer_center, optimizer_search = make_dnl_optimizer_with_center(cfg, model, center_criterion)
    start_epoch = 0

    if cfg.DNL.SEPARATE_SEARCH_SCHEDULER:
        scheduler = WarmupMultiStepLR_reinit(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                                             cfg.SOLVER.WARMUP_ITERS + cfg.DNL.MAX_SEARCH_EPOCH,
                                             cfg.SOLVER.WARMUP_METHOD, cfg.DNL.MAX_SEARCH_EPOCH,
                                             cfg.DNL.SEPARATE_SEARCH_STEPS)
    else:
        scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                                      cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)

    if cfg.DNL.LR_SCHEDULER == 'WarmupMultiStepLR':
        scheduler_search = WarmupMultiStepLR(optimizer_search, cfg.DNL.STEPS, cfg.DNL.GAMMA, cfg.DNL.WARMUP_FACTOR,
                                          cfg.DNL.WARMUP_ITERS, cfg.DNL.WARMUP_METHOD)
    else:
        scheduler_search = CosineAnnealingLR(optimizer_search, T_max=cfg.DNL.MAX_SEARCH_EPOCH + 1,
                                          eta_min=cfg.DNL.STARTPOS_LR)

    do_search(
        cfg,
        model,
        center_criterion,
        train_loader,
        val_loader,
        optimizer,
        optimizer_center,
        optimizer_search,
        scheduler,
        scheduler_search,
        loss_func,
        num_query,
        start_epoch  # add for using self trained model
    )


def main():
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument("--config_file", default="", help="path to config file", type=str)
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
        os.makedirs(output_dir)

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

    train_supernet(cfg)


if __name__ == '__main__':
    main()