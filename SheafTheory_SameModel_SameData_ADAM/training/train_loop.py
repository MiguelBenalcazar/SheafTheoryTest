import argparse
import gc

import logging
import math
from typing import Iterable

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torchmetrics.aggregation import MeanMetric
from training.grad_scaler import NativeScalerWithGradNormCount

logger = logging.getLogger(__name__)
PRINT_FREQUENCY = 50

def train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    lr_schedule: torch.torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    epoch: int,
    loss_fn: torch.nn.Module,
    args: argparse.Namespace,
):

    gc.collect()
    model.train(True)
    batch_loss = MeanMetric().to(device, non_blocking=True)
    epoch_loss = MeanMetric().to(device, non_blocking=True)

    accum_iter = args.accum_iter
    

    for data_iter_step, (samples, labels) in enumerate(data_loader):
  

        if data_iter_step % accum_iter == 0:
            optimizer.zero_grad()
            batch_loss.reset()
            if data_iter_step > 0 and args.test_run:
                break

        samples = samples.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            logits = model(samples)                # [batch_size, 10]
            loss = loss_fn(logits, labels)  # classification loss

        loss_value = loss.item()
        batch_loss.update(loss)
        epoch_loss.update(loss)     

        if not math.isfinite(loss_value):
            raise ValueError(f"Loss is {loss_value}, stopping training")

        loss /= accum_iter

        # Loss scaler applies the optimizer when update_grad is set to true.
        # Otherwise just updates the internal gradient scales
        apply_update = (data_iter_step + 1) % accum_iter == 0
        if apply_update:
            loss.backward()
            optimizer.step()
        
        
        lr = optimizer.param_groups[0]["lr"]
        if data_iter_step % PRINT_FREQUENCY == 0:
            logger.info(
                f"Epoch {epoch} [{data_iter_step}/{len(data_loader)}]: "
                f"loss = {batch_loss.compute():.4f}, lr = {lr:.6f}"
            )

    lr_schedule.step()
    return {"loss": float(epoch_loss.compute().detach().cpu())}
        
        
