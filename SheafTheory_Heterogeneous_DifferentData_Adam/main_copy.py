import datetime
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms

import torch.nn as nn
import torch.nn.functional as F
# from models.model_configs import instantiate_model
from train_arg_parser import get_args_parser

from training import distributed_mode

from training.load_and_save import load_model, save_model
from training.train_loop import train_one_epoch
from training.grad_scaler import NativeScalerWithGradNormCount as NativeScaler

from models.models import SimpleCNN, SmallVGG
from dataset_functions.datasetProcess import DatasetInformation

from optimizers.optimizer import model_optimizer, restriction_maps_optimizer


from itertools import zip_longest


logger = logging.getLogger(__name__)

lambda_reg = 0.1

        
def main(args):
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    distributed_mode.init_distributed_mode(args)

    logger.info("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    logger.info("{}".format(args).replace(", ", ",\n"))
    if distributed_mode.is_main_process():
        args_filepath = Path(args.output_dir) / "args.json"
        logger.info(f"Saving args to {args_filepath}")
        with open(args_filepath, "w") as f:
            json.dump(vars(args), f)

    device = torch.device(args.device)

    
    # fix the seed for reproducibility
    seed = args.seed + distributed_mode.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True


    logger.info(f"Initializing Dataset: {"cifar10"}")

    dataset_cifar10 = DatasetInformation(name="cifar10")
    dataset_train_1 = dataset_cifar10.dataset_Training()
    dataset_test_1 = dataset_cifar10.dataset_Testing()

   
    logger.info(f"Creating data loaders")

    logger.info("Intializing DataLoader")
    num_tasks = distributed_mode.get_world_size()
    global_rank = distributed_mode.get_rank()

    sampler_train_model1 = torch.utils.data.DistributedSampler(
        dataset_train_1, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )


    data_loader_train_1 = torch.utils.data.DataLoader(
        dataset_train_1,
        sampler=sampler_train_model1,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_test_1 = torch.utils.data.DataLoader(
        dataset_test_1,

        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

  
    model_1 = SimpleCNN(num_classes=dataset_cifar10.num_classes)
    model_1.to(device)

    model_2 = SmallVGG(num_classes=dataset_cifar10.num_classes)
    model_2.to(device)
  
    eff_batch_size = (
        args.batch_size * args.accum_iter * distributed_mode.get_world_size()
    )
    
    logger.info(f"Learning rate: {args.lr:.2e}")

    logger.info(f"Accumulate grad iterations: {args.accum_iter}")
    logger.info(f"Effective batch size: {eff_batch_size}")

    # Wrap models for distributed training if needed
    if args.distributed:
        model_cifar = torch.nn.parallel.DistributedDataParallel(
            model_1, device_ids=[args.gpu], find_unused_parameters=True
        )

    
    # Projection layers (latent_dim = 1024)
    latent_dim = 64
    P12 = torch.nn.Linear(sum(p.numel() for p in model_1.parameters()), latent_dim, bias=False)
    P21 = torch.nn.Linear(sum(p.numel() for p in model_2.parameters()), latent_dim, bias=False)

    P12.weight.requires_grad = False
    P21.weight.requires_grad = False

    torch.nn.init.xavier_uniform_(P12.weight)
    torch.nn.init.xavier_uniform_(P21.weight)

    P12.to(device)
    P21.to(device)


    # optimizer_model_1 = model_optimizer(model_1, model_2, P12, P21, lr=args.lr, lambda_reg=lambda_reg)
    # logger.info(f"Optimizer Model 1: {optimizer_model_1}")  

    # optimizer_model_2 = model_optimizer(model_2, model_1, P21, P12, lr=args.lr, lambda_reg=lambda_reg)
    # logger.info(f"Optimizer Model 2: {optimizer_model_2}")


    optimizer_model_1 = model_optimizer(model_1.parameters(), lr=args.lr, lambda_reg=lambda_reg)
    logger.info(f"Optimizer Model 1: {optimizer_model_1}")  

    optimizer_model_2 = model_optimizer(model_2.parameters(), lr=args.lr, lambda_reg=lambda_reg)
    logger.info(f"Optimizer Model 2: {optimizer_model_2}")

    restriction_updater = restriction_maps_optimizer(P12, P21, lr= args.lr, lambda_reg=lambda_reg) 
    logger.info(f"Restrictions Maps: {restriction_updater}")


    # # Choose optimizers separately
    # optimizer_model_1 = model_optimizer(model_1.parameters(), lr=args.lr)
    # logger.info(f"Optimizer: {optimizer_model_1}")

    # optimizer_model_2 = model_optimizer(model_2.parameters(), lr=args.lr)
    # logger.info(f"Optimizer: {optimizer_model_2}")

    # if args.decay_lr:
    #     lr_schedule = torch.optim.lr_scheduler.LinearLR(
    #         optimizer_cifar,
    #         total_iters=args.epochs,
    #         start_factor=1.0,
    #         end_factor=1e-8 / args.lr,
    #     )
    # else:
    #     lr_schedule = torch.optim.lr_scheduler.ConstantLR(
    #         optimizer_cifar, total_iters=args.epochs, factor=1.0
    #     )

    
    # logger.info(f"Learning-Rate Schedule: {lr_schedule}")

    loss_1 = nn.CrossEntropyLoss()
    loss_2 = nn.CrossEntropyLoss()

   



    logger.info(f"Start from {args.start_epoch} to {args.epochs} epochs")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train_1.sampler.set_epoch(epoch)

        model_1.train()
        model_2.train()
        total_loss_model_1 = 0.0
        total_loss_model_2 = 0.0
     
        for i, (samples, labels) in enumerate(data_loader_train_1):
            samples = samples.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            
          
            optimizer_model_1.zero_grad()
            outputs_1 = model_1(samples)                # [batch_size, 10]
            loss_model_1 = loss_1(outputs_1, labels)
            loss_model_1.backward()


            optimizer_model_2.zero_grad()
            outputs_2 = model_2(samples)                # [batch_size, 10]
            loss_model_2 = loss_2(outputs_2, labels)
            loss_model_2.backward()

            # Theta vectors before update
            theta1_vec = torch.cat([p.view(-1)for p in model_1.parameters()])
            theta2_vec = torch.cat([p.view(-1) for p in model_2.parameters()])
            
            optimizer_model_1.step(theta1_vec, theta2_vec, P12, P21)
            optimizer_model_2.step(theta2_vec, theta1_vec, P21, P12)


            # Theta vectors after update
            theta1_vec = torch.cat([p.view(-1)for p in model_1.parameters()])
            theta2_vec = torch.cat([p.view(-1) for p in model_2.parameters()])

            
            # update P12 and P21 in the direction of reducing discrepancy
            restriction_updater.step(theta1_vec, theta2_vec) 

            

            total_loss_model_1 += loss_model_1.item()
            total_loss_model_2 += loss_model_2.item()
            if i % 500 == 99:  # print every 100 mini-batches
                logger.info(f"[Model 1: Epoch {epoch+1}, Batch {i+1}] loss: {total_loss_model_1/100:.6f}")
                logger.info(f"[Model 2: Epoch {epoch+1}, Batch {i+1}] loss: {total_loss_model_2/100:.6f}")
                total_loss_model_1 = 0.0
                total_loss_model_2 = 0.0

    

                proj1 = P12(theta1_vec)
                proj2 = P21(theta2_vec)

                discrepancy = torch.nn.functional.mse_loss(proj1, proj2)
                logger.info(f"Discrepancy: {discrepancy.item()}")

        
        # ----- Validation phase -----
        model_1.eval()
        model_2.eval()

        with torch.no_grad():
            for i, (samples, labels) in enumerate(data_loader_test_1):
                samples = samples.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                
                outputs_1 = model_1(samples)   
                # outputs = model(inputs)
                loss_model_1 = loss_1(outputs_1, labels)

                print(loss_model_1.item())

                # _, preds = torch.max(outputs, 1)
                # running_loss += loss.item() * inputs.size(0)
                # running_corrects += torch.sum(preds == labels.data)
                # total += labels.size(0)


            


           
            
           


        
        


    


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)