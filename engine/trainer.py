# encoding: utf-8


import logging
import numpy as np
import torch
import torch.nn as nn
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage
from utils.reid_metric import R1_mAP
import wandb

global ITER
ITER = 0
global ITER_ALL
ITER_ALL = 0

def create_supervised_trainer(model, optimizer, loss_fn, loss_branch, device=None):
    """
    Factory function for creating a trainer for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        optimizer (`torch.optim.Optimizer`): the optimizer to use
        loss_fn (torch.nn loss function): the loss function to use
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.

    Returns:
        Engine: a trainer engine with supervised update function
    """
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    def _update(engine, batch):
        if loss_branch == 1:
            model.train()
            optimizer.zero_grad()
            img, target = batch
            img = img.to(device) if torch.cuda.device_count() >= 1 else img
            target = target.to(device) if torch.cuda.device_count() >= 1 else target
            score, feat = model(img)
            loss = loss_fn(score, feat, target)
            loss.backward()
            optimizer.step()

            # check the value of b
            # if b in [0,1):

            # else:
            #  get the int part of b = b - floor(b)
            #  define a function set_b() in nn.Module of model
            #  update model a = a + floor(b)
            #  define a function set_a() in nn.Module of model

            # compute acc
            acc = (score.max(1)[1] == target).float().mean()
            return loss.item(), acc.item()
        else:
            model.train()
            optimizer.zero_grad()
            img, target = batch
            img = img.to(device) if torch.cuda.device_count() >= 1 else img
            target = target.to(device) if torch.cuda.device_count() >= 1 else target

            prelogits_global, x_global, prelogits_local, x_local = model(img)
            loss_global = loss_fn(prelogits_global, x_global, target)
            loss_local = loss_fn(prelogits_local, x_local, target)
            loss = (loss_global + loss_local) / 2

            # print("Global Loss is {}, Local Loss is {}, and Center Loss is {}".format(loss_global, loss_local, center_criterion))

            loss.backward()
            optimizer.step()
            # compute acc
            acc = (prelogits_global.max(1)[1] == target).float().mean()
            return loss.item(), acc.item()

    return Engine(_update)


def create_supervised_trainer_with_center(model,
                                          center_criterion,
                                          optimizer,
                                          optimizer_center,
                                          loss_fn,
                                          cetner_loss_weight,
                                          device=None,
                                          loss_branch=1  ################### added by Yancheng
                                          ):
    """
    Factory function for creating a trainer for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        optimizer (`torch.optim.Optimizer`): the optimizer to use
        loss_fn (torch.nn loss function): the loss function to use
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.

    Returns:
        Engine: a trainer engine with supervised update function
    """
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    def _update(engine, batch):
        if loss_branch == 1:
            model.train()
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img, target = batch
            img = img.to(device) if torch.cuda.device_count() >= 1 else img
            target = target.to(device) if torch.cuda.device_count() >= 1 else target

            score, feat = model(img)

            loss = loss_fn(score, feat, target)

            # print("Total loss is {}, center loss is {}".format(loss, center_criterion(feat, target)))
            loss.backward()
            optimizer.step()

            for param in center_criterion.parameters():
                param.grad.data *= (1. / cetner_loss_weight)

            optimizer_center.step()

            # compute acc
            acc = (score.max(1)[1] == target).float().mean()
            return loss.item(), acc.item()
        else:

##################################### Added by Yancheng ################################################################
            model.train()

            optimizer.zero_grad()
            optimizer_center.zero_grad()

            img, target = batch
            img = img.to(device) if torch.cuda.device_count() >= 1 else img
            target = target.to(device) if torch.cuda.device_count() >= 1 else target

            prelogits_global, x_global, prelogits_local, x_local = model(img)
            loss_global = loss_fn(prelogits_global, x_global, target)
            loss_local = loss_fn(prelogits_local, x_local, target)
            loss = (loss_global + loss_local) / 2

            # print("Global Loss is {}, Local Loss is {}, and Center Loss is {}".format(loss_global, loss_local, center_criterion))
            loss.backward()
            optimizer.step()
            for param in center_criterion.parameters():
                param.grad.data *= (1. / cetner_loss_weight)

            optimizer_center.step()

            # compute acc
            acc = (prelogits_global.max(1)[1] == target).float().mean()
            return loss.item(), acc.item()
##################################### Added by Yancheng ################################################################

    return Engine(_update)

#
# def create_dnl_trainer_with_center(model,
#                                    center_criterion,
#                                    optimizer,
#                                    optimizer_center,
#                                    optimizer_dnl,
#                                    loss_fn,
#                                    cetner_loss_weight,
#                                    device=None,
#                                    loss_branch=1):
#     """
#     Factory function for creating a trainer for supervised models
#
#     Args:
#         model (`torch.nn.Module`): the model to train
#         optimizer (`torch.optim.Optimizer`): the optimizer to use
#         loss_fn (torch.nn loss function): the loss function to use
#         device (str, optional): device type specification (default: None).
#             Applies to both model and batches.
#
#     Returns:
#         Engine: a trainer engine with supervised update function
#     """
#     if device:
#         if torch.cuda.device_count() > 1:
#             model = nn.DataParallel(model)
#         model.to(device)
#
#     def _update(engine, batch):
#
#         model.train()
#
#         optimizer.zero_grad()
#         optimizer_dnl.zero_grad()
#         optimizer_center.zero_grad()
#
#         img, target = batch
#         img = img.to(device) if torch.cuda.device_count() >= 1 else img
#         target = target.to(device) if torch.cuda.device_count() >= 1 else target
#
#         score, feat = model(img)
#         loss = loss_fn(score, feat, target)
#
#         # print("Total loss is {}, center loss is {}".format(loss, center_criterion(feat, target)))
#         loss.backward()
#
#         optimizer.step()
#
#         optimizer_dnl.step()
#
#         for param in center_criterion.parameters():
#             param.grad.data *= (1. / cetner_loss_weight)
#
#         optimizer_center.step()
#
#         num_nl = 4
#         ranges = [24, 32, 64, 320]
#
#         for key, value in model.named_parameters():
#             if "startpos_dec" in key:
#                 for i in range(1, num_nl + 1):
#                     name = "non_local_" + str(i)
#                     if name in key:
#                         if 0 < value <= 1:
#                             last_dec_copy = value.data.cpu().numpy()
#                             # print('int of ' + name + ' remains the same in this iter, dec is' + str(last_dec_copy))
#                             continue
#                         else:
#                             # print('---------------------------Starting a floor processing at ' + name)
#                             last_dec_copy = value.data.cpu().numpy()
#                             # print('dec before flooring ' + str(last_dec_copy))
#                             dec_floor = np.floor(last_dec_copy)
#                             last_int = eval('model.base.non_local_' + str(i) + '.ch_startpos_int')
#                             last_int_copy = last_int.data.cpu().numpy()
#                             new_int = last_int_copy + dec_floor
#                             int_range = ranges[i-1]
#                             # print('int before flooring ' + str(last_int_copy))
#                             new_int = np.mod(int(new_int), int_range)
#                             # print('int after flooring ' + str(new_int))
#
#                             model.base._set_ch_startpos_int(new_int, i)
#                             new_dec = last_dec_copy - dec_floor
#                             # print('dec after flooring ' + str(new_dec))
#                             model.base._set_ch_startpos_dec(new_dec, i)
#                             # print('---------------------------Finishing a floor processing at ' + name)
#
#         minimum_length_ratio = 0.25
#
#         for key, value in model.named_parameters():
#             if "length_dec" in key:
#                 for i in range(1, num_nl + 1):
#                     name = "non_local_" + str(i)
#                     if name in key:
#                         if 0 < value <= 1:
#                             continue
#                         else:
#                             last_dec_copy = value.data.cpu().numpy()
#                             dec_floor = np.floor(last_dec_copy)
#                             last_int = eval('model.base.non_local_' + str(i) + '.ch_length_int')
#                             last_int_copy = last_int.data.cpu().numpy()
#                             new_int = last_int_copy + dec_floor
#                             int_range = ranges[i-1]
#                             if new_int >= int_range:
#                                 new_int = int_range - 1
#                                 new_dec = 1.0
#                                 model.base._set_ch_length_int(new_int, i)
#                                 model.base._set_ch_length_dec(new_dec, i)
#                             elif new_int < int_range * minimum_length_ratio:
#                                 new_int = int_range * minimum_length_ratio
#                                 new_dec = 0.0
#                                 model.base._set_ch_length_int(new_int, i)
#                                 model.base._set_ch_length_dec(new_dec, i)
#                             else:
#                                 model.base._set_ch_length_int(new_int, i)
#                                 new_dec = last_dec_copy - dec_floor
#                                 model.base._set_ch_length_dec(new_dec, i)
#
#
#                 # else:
#                 #     print('---------------------------Starting a floor processing---------------------------')
#                 #     last_dec_copy = value.data.cpu().numpy()
#                 #
#                 #     print('dec before flooring ' + str(last_dec_copy))
#                 #
#                 #     dec_floor = np.floor(last_dec_copy)
#                 #
#                 #     last_int = model.base._get_NL1_ch_startpos_int()
#                 #
#                 #     last_int_copy = last_int.data.cpu().numpy()
#                 #     print('int before flooring ' + str(last_int_copy))
#                 #
#                 #     new_int = last_int_copy + dec_floor
#                 #     int_range = [12]    #TODO: define as parameter
#                 #     new_int = np.mod(int(new_int), int_range)
#                 #
#                 #     print('int after flooring ' + str(new_int))
#                 #     model.base._set_NL1_ch_startpos_int(new_int)
#                 #
#                 #     new_dec = last_dec_copy - dec_floor
#                 #
#                 #     print('dec after flooring ' + str(new_dec))
#                 #     model.base._set_NL1_ch_startpos_dec(new_dec)
#                 #     print('---------------------------Finishing a floor processing---------------------------')
#
#         # compute acc
#         acc = (score.max(1)[1] == target).float().mean()
#         return loss.item(), acc.item()
#
#     return Engine(_update)

def create_supervised_evaluator(model,
                                metrics,
                                device=None,
                                loss_branch=1
                                ):
    """
    Factory function for creating an evaluator for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        metrics (dict of str - :class:`ignite.metrics.Metric`): a map of metric names to Metrics
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
    Returns:
        Engine: an evaluator engine with supervised inference function
    """
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    def _inference(engine, batch):

        model.eval()

        with torch.no_grad():

            data, pids, camids = batch

            data = data.to(device) if torch.cuda.device_count() >= 1 else data

            feat = model(data)

            return feat, pids, camids

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_fn,
        num_query,
        start_epoch
):
    ##################################### Added by Yancheng ################################################################
    multi_loss = cfg.MODEL.LOSS_BRANCH
    ##################################### Added by Yancheng ################################################################
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    output_dir = cfg.OUTPUT_DIR
    device = cfg.MODEL.DEVICE
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("reid_baseline.train")
    logger.info("Start training")

    ##################################### Added by Yancheng ################################################################
    trainer = create_supervised_trainer(
        model,
        optimizer,
        loss_fn,
        loss_branch=multi_loss,
        device=device)
    ##################################### Added by Yancheng ################################################################

    evaluator = create_supervised_evaluator(model, metrics={'r1_mAP': R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)}, device=device)
    checkpointer = ModelCheckpoint(output_dir, cfg.MODEL.NAME, checkpoint_period, n_saved=10, require_empty=False)
    timer = Timer(average=True)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'model': model,
                                                                     'optimizer': optimizer})
    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    # average metric to attach on trainer
    RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'avg_loss')
    RunningAverage(output_transform=lambda x: x[1]).attach(trainer, 'avg_acc')

    @trainer.on(Events.STARTED)
    def start_training(engine):
        engine.state.epoch = start_epoch

    @trainer.on(Events.EPOCH_STARTED)
    def adjust_learning_rate(engine):
        scheduler.step()

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        global ITER
        ITER += 1

        if ITER % log_period == 0:
            logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                        .format(engine.state.epoch, ITER, len(train_loader),
                                engine.state.metrics['avg_loss'], engine.state.metrics['avg_acc'],
                                scheduler.get_lr()[0]))
        if len(train_loader) == ITER:
            ITER = 0

    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        logger.info('Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'
                    .format(engine.state.epoch, timer.value() * timer.step_count,
                            train_loader.batch_size / timer.value()))
        logger.info('-' * 10)
        ############################ WDB
        if cfg.TEST.IF_WDB:
            wandb.log({"training loss": engine.state.metrics['avg_loss']}, step=engine.state.epoch)
            wandb.log({"accuracy on training": engine.state.metrics['avg_acc']}, step=engine.state.epoch)
        timer.reset()

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        if engine.state.epoch % eval_period == 0:
            evaluator.run(val_loader)
            cmc, mAP = evaluator.state.metrics['r1_mAP']
            logger.info("Validation Results - Epoch: {}".format(engine.state.epoch))
            logger.info("mAP: {:.1%}".format(mAP))
            ############################ WDB
            if cfg.TEST.IF_WDB:
                wandb.log({"test mAp": mAP}, step=engine.state.epoch)
                wandb.log({"test Rank 1": cmc[0]}, step=engine.state.epoch)
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    trainer.run(train_loader, max_epochs=epochs)


def do_train_with_center(
        cfg,
        model,
        center_criterion,
        train_loader,
        val_loader,
        optimizer,
        optimizer_center,
        scheduler,
        loss_fn,
        num_query,
        start_epoch
):
    multi_loss = cfg.MODEL.LOSS_BRANCH
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    output_dir = cfg.OUTPUT_DIR
    device = cfg.MODEL.DEVICE
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("reid_baseline.train")
    logger.info("Start training")

    trainer = create_supervised_trainer_with_center(model,
                                                    center_criterion,
                                                    optimizer,
                                                    optimizer_center,
                                                    loss_fn,
                                                    cfg.SOLVER.CENTER_LOSS_WEIGHT,
                                                    device=device,
                                                    loss_branch=multi_loss  # added by Yancheng
                                                    )

    evaluator = create_supervised_evaluator(model, metrics={'r1_mAP': R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)}, device=device)

    checkpointer = ModelCheckpoint(output_dir, cfg.MODEL.NAME, checkpoint_period, n_saved=10, require_empty=False)

    timer = Timer(average=True)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'model': model,
                                                                     'optimizer': optimizer,
                                                                     'center_param': center_criterion,
                                                                     'optimizer_center': optimizer_center})

    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    # average metric to attach on trainer
    RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'avg_loss')
    RunningAverage(output_transform=lambda x: x[1]).attach(trainer, 'avg_acc')

    @trainer.on(Events.ITERATION_STARTED)
    # WDB operations---------------------- Iter
    def print_position(engine):
        global ITER_ALL
        ITER_ALL += 1
        if cfg.TEST.IF_WDB:
            if cfg.TEST.WDB_PRINT_ITER:
                if cfg.TEST.WDB_PRINT_POSITION:
                    num_pos = len(model.base.conv6[0].conv3[0].position)
                    for i in range(0, num_pos):
                        names = 'position' + str(i)
                        wandb.log({str(names): model.base.conv6[0].conv3[0].position[i].cpu().data.numpy()},
                                  step=ITER_ALL)

    @trainer.on(Events.ITERATION_COMPLETED)
    # WDB operations---------------------- Iter
    def wandb_training_log_iter(engine):
        global ITER_ALL
        if cfg.TEST.IF_WDB:
            if cfg.TEST.WDB_PRINT_ITER:
                if ITER_ALL > 0:
                    wandb.log({"training loss": engine.state.metrics['avg_loss']}, step=ITER_ALL)
                    wandb.log({"accuracy on training": engine.state.metrics['avg_acc']}, step=ITER_ALL)

    @trainer.on(Events.STARTED)
    def start_training(engine):
        engine.state.epoch = start_epoch

    @trainer.on(Events.EPOCH_STARTED)
    def adjust_learning_rate(engine):
        scheduler.step()

    @trainer.on(Events.EPOCH_STARTED)
    # WDB operations---------------------- Iter
    def freeze_position_param(engine):
        if cfg.TEST.IF_WDB:
            if cfg.TEST.WDB_PRINT_ITER:
                if cfg.TEST.WDB_PRINT_POSITION:

                    if engine.state.epoch == cfg.SOLVER.FREEZE_POS:
                        for names, modules in model.named_parameters():
                            if "position" in names:
                                modules.requires_grad = False



    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        global ITER
        ITER += 1

        if ITER % log_period == 0:
            logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                        .format(engine.state.epoch, ITER, len(train_loader),
                                engine.state.metrics['avg_loss'], engine.state.metrics['avg_acc'],
                                scheduler.get_lr()[0]))
        if len(train_loader) == ITER:
            ITER = 0

    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        logger.info('Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'
                    .format(engine.state.epoch, timer.value() * timer.step_count,
                            train_loader.batch_size / timer.value()))
        logger.info('-' * 10)

        # WDB operations---------------------- Epoch
        if cfg.TEST.IF_WDB:
            if not cfg.TEST.WDB_PRINT_ITER:
                wandb.log({"training loss": engine.state.metrics['avg_loss']}, step=engine.state.epoch)
                wandb.log({"accuracy on training": engine.state.metrics['avg_acc']}, step=engine.state.epoch)
        timer.reset()

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        if engine.state.epoch % eval_period == 0:
            evaluator.run(val_loader)
            cmc, mAP = evaluator.state.metrics['r1_mAP']
            logger.info("Validation Results - Epoch: {}".format(engine.state.epoch))
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

            # WDB operations---------------------- Epoch
            if cfg.TEST.IF_WDB:
                if not cfg.TEST.WDB_PRINT_ITER:
                    wandb.log({"test mAp": mAP}, step=engine.state.epoch)
                    wandb.log({"test Rank 1": cmc[0]}, step=engine.state.epoch)

    @trainer.on(Events.EPOCH_STARTED)
    def lock_network(engine):
        if cfg.SOLVER.TWO_STEP:
            epoch_num = engine.state.epoch
            if epoch_num <= cfg.SOLVER.TWO_STEP_MAX:
                # lock search options
                for key, value in model.named_parameters():
                    if 'base' in key:
                        if 'ch_startpos_dec' in key:
                            value.requires_grad = False
                        if 'ch_length_dec' in key:
                            value.requires_grad = False
                # lock network parameters
                for key, value in model.named_parameters():
                    if 'base' in key:
                        if 'non_local' not in key:
                            value.requires_grad = False



    @trainer.on(Events.EPOCH_STARTED)
    def unlock_network(engine):
        if cfg.SOLVER.TWO_STEP:
            epoch_num = engine.state.epoch
            if epoch_num == cfg.SOLVER.TWO_STEP_MAX + 1:
                # unlock network parameters
                for key, value in model.named_parameters():
                    if 'base' in key:
                        if 'non_local' not in key:
                            value.requires_grad = True


    trainer.run(train_loader, max_epochs=epochs)

def create_dnl_trainer_with_center(model,
                                   center_criterion,
                                   optimizer,
                                   optimizer_center,
                                   optimizer_dnl,
                                   loss_fn,
                                   cetner_loss_weight,
                                   cfg,
                                   device=None,
                                   loss_branch=1):
    """
    Factory function for creating a trainer for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        optimizer (`torch.optim.Optimizer`): the optimizer to use
        loss_fn (torch.nn loss function): the loss function to use
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.

    Returns:
        Engine: a trainer engine with supervised update function
    """
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    def _update(engine, batch):

        model.train()

        optimizer.zero_grad()
        optimizer_dnl.zero_grad()
        optimizer_center.zero_grad()

        img, target = batch
        img = img.to(device) if torch.cuda.device_count() >= 1 else img
        target = target.to(device) if torch.cuda.device_count() >= 1 else target

        score, feat = model(img)
        loss = loss_fn(score, feat, target)

        # print("Total loss is {}, center loss is {}".format(loss, center_criterion(feat, target)))
        loss.backward()

        # for key, value in model.named_parameters():
        #     if 'ch_length_dec' in key:
        #         print('-->name:', key, '-->requires_grad:', value.requires_grad, ' -->grad_value:', value.grad)

        optimizer.step()

        optimizer_dnl.step()

        for param in center_criterion.parameters():
            param.grad.data *= (1. / cetner_loss_weight)

        optimizer_center.step()

        num_nl = 4
        ranges = [24, 32, 64, 320]

        if cfg.DNL.LEARN_STARTPOS:
            for key, value in model.named_parameters():
                if "startpos_dec" in key:
                    for i in range(1, num_nl + 1):
                        name = "non_local_" + str(i)
                        if name in key:
                            if 0 <= value < 1:
                                last_dec_copy = value.data.cpu().numpy()
                                # print('int of ' + name + ' remains the same in this iter, dec is' + str(last_dec_copy))
                                continue
                            else:
                                # print('---------------------------Starting a floor processing at ' + name)
                                last_dec_copy = value.data.cpu().numpy()
                                # print('dec before flooring ' + str(last_dec_copy))
                                dec_floor = np.floor(last_dec_copy)
                                last_int = eval('model.base.non_local_' + str(i) + '.ch_startpos_int')
                                last_int_copy = last_int.data.cpu().numpy()
                                new_int = last_int_copy + dec_floor
                                int_range = ranges[i - 1]
                                # print('int before flooring ' + str(last_int_copy))
                                new_int = np.mod(int(new_int), int_range)
                                # print('int after flooring ' + str(new_int))
                                model.base._set_ch_startpos_int(new_int, i)
                                new_dec = last_dec_copy - dec_floor
                                # print('dec after flooring ' + str(new_dec))
                                model.base._set_ch_startpos_dec(new_dec, i)
                                # print('---------------------------Finishing a floor processing at ' + name)

        if cfg.DNL.LEARN_LENGTH:
            minimum_length_ratio = 0.25

            for key, value in model.named_parameters():
                if "length_dec" in key:
                    for i in range(1, num_nl + 1):
                        name = "non_local_" + str(i)
                        if name in key:
                            if 0 <= value < 1:
                                continue
                            else:
                                last_dec_copy = value.data.cpu().numpy()
                                dec_floor = np.floor(last_dec_copy)
                                last_int = eval('model.base.non_local_' + str(i) + '.ch_length_int')
                                last_int_copy = last_int.data.cpu().numpy()
                                new_int = int(last_int_copy + dec_floor)
                                int_range = ranges[i - 1]

                                if new_int >= int_range:
                                    # print('condition 1 at ' + name)
                                    new_int = int(int_range - 1)
                                    new_dec = last_dec_copy - dec_floor
                                    model.base._set_ch_length_int(new_int, i)
                                    model.base._set_ch_length_dec(new_dec, i)

                                elif new_int < int_range * minimum_length_ratio:
                                    # print('condition 2 at ' + name)
                                    new_int = int(int_range * minimum_length_ratio)
                                    new_dec = last_dec_copy - dec_floor
                                    model.base._set_ch_length_int(new_int, i)
                                    model.base._set_ch_length_dec(new_dec, i)

                                else:
                                    model.base._set_ch_length_int(new_int, i)
                                    new_dec = last_dec_copy - dec_floor
                                    model.base._set_ch_length_dec(new_dec, i)


                # else:
                #     print('---------------------------Starting a floor processing---------------------------')
                #     last_dec_copy = value.data.cpu().numpy()
                #
                #     print('dec before flooring ' + str(last_dec_copy))
                #
                #     dec_floor = np.floor(last_dec_copy)
                #
                #     last_int = model.base._get_NL1_ch_startpos_int()
                #
                #     last_int_copy = last_int.data.cpu().numpy()
                #     print('int before flooring ' + str(last_int_copy))
                #
                #     new_int = last_int_copy + dec_floor
                #     int_range = [12]    #TODO: define as parameter
                #     new_int = np.mod(int(new_int), int_range)
                #
                #     print('int after flooring ' + str(new_int))
                #     model.base._set_NL1_ch_startpos_int(new_int)
                #
                #     new_dec = last_dec_copy - dec_floor
                #
                #     print('dec after flooring ' + str(new_dec))
                #     model.base._set_NL1_ch_startpos_dec(new_dec)
                #     print('---------------------------Finishing a floor processing---------------------------')

        # compute acc
        acc = (score.max(1)[1] == target).float().mean()
        return loss.item(), acc.item()

    return Engine(_update)


def do_train_dnl_with_center(
        cfg,
        model,
        center_criterion,
        train_loader,
        val_loader,
        optimizer,
        optimizer_center,
        optimizer_dnl,
        scheduler,
        scheduler_dnl,
        loss_fn,
        num_query,
        start_epoch
):
    multi_loss = cfg.MODEL.LOSS_BRANCH
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    output_dir = cfg.OUTPUT_DIR
    device = cfg.MODEL.DEVICE
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("reid_baseline.train")
    logger.info("Start training")

    trainer = create_dnl_trainer_with_center(model,
                                             center_criterion,
                                             optimizer,
                                             optimizer_center,
                                             optimizer_dnl,
                                             loss_fn,
                                             cfg.SOLVER.CENTER_LOSS_WEIGHT,
                                             cfg,
                                             device=device,
                                             loss_branch=multi_loss
                                             )

    evaluator = create_supervised_evaluator(model, metrics={'r1_mAP': R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)}, device=device)

    checkpointer = ModelCheckpoint(output_dir, cfg.MODEL.NAME, checkpoint_period, n_saved=10, require_empty=False)

    timer = Timer(average=True)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'model': model,
                                                                     'optimizer': optimizer,
                                                                     'center_param': center_criterion,
                                                                     'optimizer_center': optimizer_center})

    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    # average metric to attach on trainer
    RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'avg_loss')
    RunningAverage(output_transform=lambda x: x[1]).attach(trainer, 'avg_acc')

    @trainer.on(Events.ITERATION_COMPLETED)
    # WDB operations---------------------- Iter
    def print_position(engine):
        global ITER_ALL
        ITER_ALL += 1
        if cfg.TEST.IF_WDB:
            if cfg.TEST.WDB_PRINT_ITER:
                if cfg.TEST.WDB_PRINT_POSITION:

                    if cfg.DNL.LEARN_STARTPOS:
                        num_pos = 4
                        for i in range(1, num_pos + 1):
                            a = model.base.non_local_1.ch_startpos_int
                            dnl_int = eval('model.base.non_local_' + str(i) + '.ch_startpos_int')
                            names = 'model.base.non_local_' + str(i) + '.ch_startpos_int'
                            # wandb.log({str(names): dnl_int.cpu().data.numpy()}, step=ITER_ALL)
                            dnl_dec = eval('model.base.non_local_' + str(i) + '.ch_startpos_dec')
                            names = 'model.base.non_local_' + str(i) + '.ch_startpos_dec'
                            # wandb.log({str(names): dnl_dec.cpu().data.numpy()}, step=ITER_ALL)
                            startpos = dnl_dec.cpu().data.numpy() + dnl_int.cpu().data.numpy()
                            names = 'model.base.non_local_' + str(i) + '.ch_startpos'
                            wandb.log({str(names): startpos}, step=ITER_ALL)

                    if cfg.DNL.LEARN_LENGTH:
                        num_pos = 4
                        for i in range(1, num_pos + 1):
                            a = model.base.non_local_1.ch_length_int
                            dnl_int = eval('model.base.non_local_' + str(i) + '.ch_length_int')
                            names = 'model.base.non_local_' + str(i) + '.ch_length_int'
                            # wandb.log({str(names): dnl_int.cpu().data.numpy()}, step=ITER_ALL)
                            dnl_dec = eval('model.base.non_local_' + str(i) + '.ch_length_dec')
                            names = 'model.base.non_local_' + str(i) + '.ch_length_dec'
                            # wandb.log({str(names): dnl_dec.cpu().data.numpy()}, step=ITER_ALL)
                            length = dnl_dec.cpu().data.numpy() + dnl_int.cpu().data.numpy()
                            names = 'model.base.non_local_' + str(i) + '.ch_length'
                            wandb.log({str(names): length}, step=ITER_ALL)

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        global ITER
        ITER += 1

        if ITER % log_period == 0:
            logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                        .format(engine.state.epoch, ITER, len(train_loader),
                                engine.state.metrics['avg_loss'], engine.state.metrics['avg_acc'],
                                scheduler.get_lr()[0]))
        if len(train_loader) == ITER:
            ITER = 0

    @trainer.on(Events.ITERATION_COMPLETED)
    # WDB operations---------------------- Iter
    def wandb_training_log_iter(engine):
        global ITER_ALL
        if cfg.TEST.IF_WDB:
            if cfg.TEST.WDB_PRINT_ITER:
                if ITER_ALL > 0:
                    wandb.log({"training loss": engine.state.metrics['avg_loss']}, step=ITER_ALL)
                    wandb.log({"accuracy on training": engine.state.metrics['avg_acc']}, step=ITER_ALL)

    @trainer.on(Events.EPOCH_STARTED)
    def adjust_learning_rate(engine):
        scheduler.step()
        scheduler_dnl.step()

    @trainer.on(Events.EPOCH_STARTED)
    def lock_arch_parameter(engine):
        if cfg.DNL.SEPARATE_TRAINING:
            epoch_num = engine.state.epoch

            if epoch_num < cfg.DNL.START_SEARCH:
                for key, value in model.named_parameters():
                    if 'base' in key:
                        if 'ch_startpos_dec' in key:
                            value.requires_grad = False
                        elif 'ch_length_dec' in key:
                            value.requires_grad = False
                        else:
                            if 'ch_startpos_int' not in key:
                                if 'ch_length_int' not in key:
                                    value.requires_grad = True

            elif epoch_num <= cfg.DNL.MAX_SEARCH_EPOCH:
                mod_num = np.mod(epoch_num, cfg.DNL.ITER_EPOCH_NUM)
                if mod_num < int(cfg.DNL.ITER_EPOCH_NUM * cfg.DNL.WEIGHT_FACTOR):
                    for key, value in model.named_parameters():
                        if 'base' in key:
                            if 'ch_startpos_dec' in key:
                                value.requires_grad = False
                            elif 'ch_length_dec' in key:
                                value.requires_grad = False
                            else:
                                if 'ch_startpos_int' not in key:
                                    if 'ch_length_int' not in key:
                                        value.requires_grad = True

    @trainer.on(Events.EPOCH_STARTED)
    def lock_regular_parameter(engine):
        if cfg.DNL.SEPARATE_TRAINING:
            epoch_num = engine.state.epoch

            if epoch_num >= cfg.DNL.START_SEARCH:

                if epoch_num <= cfg.DNL.MAX_SEARCH_EPOCH:
                    mod_num = np.mod(epoch_num, cfg.DNL.ITER_EPOCH_NUM)
                    if mod_num >= int(cfg.DNL.ITER_EPOCH_NUM * cfg.DNL.WEIGHT_FACTOR):
                        for key, value in model.named_parameters():
                            if 'base' in key:
                                if 'ch_startpos_dec' in key:
                                    value.requires_grad = True
                                elif 'ch_length_dec' in key:
                                    value.requires_grad = True
                                else:
                                    value.requires_grad = False



    @trainer.on(Events.EPOCH_STARTED)
    def finish_seperate_training(engine):
        if cfg.DNL.SEPARATE_TRAINING:
            epoch_num = engine.state.epoch
            if epoch_num == cfg.DNL.MAX_SEARCH_EPOCH + 1:
                for key, value in model.named_parameters():
                    if 'base' in key:
                        if 'ch_startpos_dec' in key:
                            value.requires_grad = False
                        elif 'ch_length_dec' in key:
                            value.requires_grad = False
                        else:
                            if 'ch_startpos_int' not in key:
                                if 'ch_length_int' not in key:
                                    value.requires_grad = True


    @trainer.on(Events.EPOCH_STARTED)
    def init_after_search(engine):
        if cfg.DNL.SEPARATE_TRAINING:
            if cfg.DNL.INIT_AFTER_SEARCH:
                epoch_num = engine.state.epoch
                if epoch_num == cfg.DNL.MAX_SEARCH_EPOCH + 1:
                    a = model.base.non_local_1.ch_length_int
                    model.base._init_params()

        elif cfg.DNL.SEPARATE_TRAINING_ITER:
            if cfg.DNL.INIT_AFTER_SEARCH:
                epoch_num = engine.state.epoch
                if epoch_num == cfg.DNL.MAX_SEARCH_EPOCH + 1:
                    a = model.base.non_local_1.ch_length_int
                    model.base._init_params()


    @trainer.on(Events.ITERATION_STARTED)
    def lock_arch_parameter_iter(engine):
        global ITER_ALL
        if cfg.DNL.SEPARATE_TRAINING_ITER:
            epoch_num = engine.state.epoch
            if epoch_num <= cfg.DNL.MAX_SEARCH_EPOCH:
                if np.mod(ITER_ALL, cfg.DNL.ITER_EPOCH_NUM) < int(cfg.DNL.ITER_EPOCH_NUM * cfg.DNL.WEIGHT_FACTOR):
                    for key, value in model.named_parameters():
                        if 'base' in key:
                            if 'ch_startpos_dec' in key:
                                value.requires_grad = False
                            elif 'ch_length_dec' in key:
                                value.requires_grad = False
                            else:
                                if 'ch_startpos_int' not in key:
                                    if 'ch_length_int' not in key:
                                        value.requires_grad = True

    @trainer.on(Events.ITERATION_STARTED)
    def lock_regular_parameter_iter(engine):
        global ITER_ALL
        if cfg.DNL.SEPARATE_TRAINING_ITER:
            epoch_num = engine.state.epoch
            if epoch_num <= cfg.DNL.MAX_SEARCH_EPOCH:
                if np.mod(ITER_ALL, cfg.DNL.ITER_EPOCH_NUM) >= int(cfg.DNL.ITER_EPOCH_NUM * cfg.DNL.WEIGHT_FACTOR):
                    for key, value in model.named_parameters():
                        if 'zero' in key:
                            continue
                        if 'base' in key:
                            if 'ch_startpos_dec' in key:
                                value.requires_grad = True
                            elif 'ch_length_dec' in key:
                                value.requires_grad = True
                            else:
                                value.requires_grad = False

    @trainer.on(Events.EPOCH_STARTED)
    def finish_seperate_training_iter(engine):
        if cfg.DNL.SEPARATE_TRAINING_ITER:
            epoch_num = engine.state.epoch
            if epoch_num == cfg.DNL.MAX_SEARCH_EPOCH + 1:
                for key, value in model.named_parameters():
                    if 'base' in key:
                        if 'ch_startpos_dec' in key:
                            value.requires_grad = False
                        elif 'ch_length_dec' in key:
                            value.requires_grad = False
                        else:
                            if 'ch_startpos_int' not in key:
                                if 'ch_length_int' not in key:
                                    value.requires_grad = True

    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        logger.info('Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'
                    .format(engine.state.epoch, timer.value() * timer.step_count,
                            train_loader.batch_size / timer.value()))
        logger.info('-' * 10)

        # # WDB operations---------------------- Epoch
        # if cfg.TEST.IF_WDB:
        #     if not cfg.TEST.WDB_PRINT_ITER:
        #         wandb.log({"training loss": engine.state.metrics['avg_loss']}, step=engine.state.epoch)
        #         wandb.log({"accuracy on training": engine.state.metrics['avg_acc']}, step=engine.state.epoch)
        timer.reset()

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        if engine.state.epoch % eval_period == 0:
            evaluator.run(val_loader)
            cmc, mAP = evaluator.state.metrics['r1_mAP']
            logger.info("Validation Results - Epoch: {}".format(engine.state.epoch))
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

            # # WDB operations---------------------- Epoch
            # if cfg.TEST.IF_WDB:
            #     if not cfg.TEST.WDB_PRINT_ITER:
            #         wandb.log({"test mAp": mAP}, step=engine.state.epoch)
            #         wandb.log({"test Rank 1": cmc[0]}, step=engine.state.epoch)


    @trainer.on(Events.STARTED)
    def start_training(engine):
        engine.state.epoch = start_epoch

    trainer.run(train_loader, max_epochs=epochs)

