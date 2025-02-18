# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch
import utils
from losses import DistillationLoss
from timm.data import Mixup
from timm.utils import ModelEma, accuracy


def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, args = None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    
    if args.cosub:
        criterion = torch.nn.BCEWithLogitsLoss()
        
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
            
        if args.cosub:
            samples = torch.cat((samples,samples),dim=0)
            
        if args.bce_loss:
            targets = targets.gt(0.0).type(targets.dtype)
         
        with torch.cuda.amp.autocast():
            outputs = model(samples)
            if not args.cosub:
                loss = criterion(samples, outputs, targets)
            else:
                outputs = torch.split(outputs, outputs.shape[0]//2, dim=0)
                loss = 0.25 * criterion(outputs[0], targets) 
                loss = loss + 0.25 * criterion(outputs[1], targets) 
                loss = loss + 0.25 * criterion(outputs[0], outputs[1].detach().sigmoid())
                loss = loss + 0.25 * criterion(outputs[1], outputs[0].detach().sigmoid()) 

        ####################################### F A C T #######################################
        if args.l2_weight > 0:

            # take the l2-norm from the last layer
            # [cls, patches, regs]
            discard_tokens = args.num_registers
            # not that neat but works both locally and on the cluster
            if discard_tokens > 0:
                try:
                    num_blocks = model.module.get_num_layers()-1
                    final_output = model.module.block_output[f"block{num_blocks}"][:, 1:-discard_tokens]
                    # final_output = model.module.block_output['final'][:, 1:-discard_tokens]
                except:
                    num_blocks = model.get_num_layers()-1
                    final_output = model.block_output[f"block{num_blocks}"][:, 1:-discard_tokens]
                    # final_output = model.block_output['final'][:, 1:-discard_tokens]
            else:
                try:
                    num_blocks = model.module.get_num_layers()-1
                    final_output = model.module.block_output[f"block{num_blocks}"][:, 1:]
                except:
                    num_blocks = model.get_num_layers()-1
                    final_output = model.block_output[f"block{num_blocks}"][:, 1:]
            
            output_norms = final_output.norm(dim=-1)
            if args.l2_decay:
                l2_norm_loss = args.l2_weight / (epoch + 1) * output_norms.mean()
            elif args.exp_decay:
                l2_norm_loss = args.l2_weight ** ((epoch // 10) + 1) * output_norms.mean()
            elif args.exp_soft_decay:
                l2_norm_loss = args.l2_weight ** ((epoch / 10) + 1) * output_norms.mean()
            elif args.cos_decay:
                l2_norm_loss = args.l2_weight * math.cos(math.pi * (epoch % 10) / 20) * output_norms.mean()
            elif args.step_decay:
                if epoch < 10:
                    l2_norm_loss = args.l2_weight * output_norms.mean()
                else:
                    l2_norm_loss = 0 * output_norms.mean()
            else:
                l2_norm_loss = args.l2_weight * output_norms.mean()
            print("*"*20, "Cross-entropy loss: ", loss.item())
            print("*"*20, "L2-norm loss: ", l2_norm_loss.item())
            loss = loss + l2_norm_loss
        else:
            print("*"*20, "Cross-entropy loss: ", loss.item())

        ######################################################################################
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        # only if cuda is available
        if device == torch.device('cuda'):
            torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
