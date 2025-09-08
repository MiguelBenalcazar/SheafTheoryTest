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

from models.models import RMNIST_CNN, SmallVGG, SimpleMLP, SimpleCNN
from dataset_functions.datasetProcess import DatasetInformation

from optimizers.optimizer import model_optimizer, restriction_maps_optimizer, restriction_maps_optimizer_P12, restriction_maps_optimizer_P21

from torch.utils.tensorboard import SummaryWriter
from utils.save_load import save_checkpoint    

import random

logger = logging.getLogger(__name__)


# test similar datasets different models
        
def main(args):
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    distributed_mode.init_distributed_mode(args)


    # Are you setting these identically?
    torch.manual_seed(42)
    torch.cuda.manual_seed(42) 
    np.random.seed(42)
    random.seed(42)
    torch.backends.cudnn.deterministic = True

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


    logger.info(f"Initializing Dataset: {"cifar10"} and {"cifar10"}")


    # LOAD DATASETS
    dataset_cifar10 = DatasetInformation(name="cifar10")
    dataset_train_1 = dataset_cifar10.dataset_Training()
    dataset_test_1 = dataset_cifar10.dataset_Testing()

    dataset_r_mnist = DatasetInformation(name="cifar10")
    dataset_train_2 = dataset_r_mnist.dataset_Training()
    dataset_test_2 = dataset_r_mnist.dataset_Testing()


   
    logger.info(f"Creating data loaders")

    logger.info("Intializing DataLoader")
    num_tasks = distributed_mode.get_world_size()
    global_rank = distributed_mode.get_rank()

    sampler_train_model1 = torch.utils.data.DistributedSampler(
        dataset_train_1, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )

    sampler_train_model2 = torch.utils.data.DistributedSampler(
        dataset_train_2, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )

    #DATALOADERS 
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
        drop_last=False,
    )

    data_loader_train_2 = torch.utils.data.DataLoader(
        dataset_train_2,
        sampler=sampler_train_model2,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_test_2 = torch.utils.data.DataLoader(
        dataset_test_2,

        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

  
    model_1 = SmallVGG(num_classes=dataset_cifar10.num_classes)
    model_1.to(device)

    model_2 = SmallVGG(num_classes=dataset_r_mnist.num_classes)
    model_2.to(device)
  
    eff_batch_size = (
        args.batch_size * args.accum_iter * distributed_mode.get_world_size()
    )
    
    logger.info(f"Learning rate model: {args.lr}")
    logger.info(f"Learning rate layer: {args.lr_layer}")
    logger.info(f"Learning rate restriction map: {args.lr_restriction_maps}")
    logger.info(f"Lambda (restriction map): {args.lambda_reg}")

    logger.info(f"Accumulate grad iterations: {args.accum_iter}")
    logger.info(f"Effective batch size: {eff_batch_size}")

    # Wrap models for distributed training if needed
    if args.distributed:
        model_cifar = torch.nn.parallel.DistributedDataParallel(
            model_1, device_ids=[args.gpu], find_unused_parameters=True
        )
    
    logger.info(f" Model dimension 1: {model_1.fc1.weight.shape}, Model dimension 2: {model_2.fc1.weight.shape}")

    P12 = torch.nn.Linear(  model_1.fc1.weight.shape[1], args.latent_dim,bias=False)
    P21 = torch.nn.Linear( model_2.fc1.weight.shape[1], args.latent_dim, bias=False)
    P12.weight.requires_grad = False
    P21.weight.requires_grad = False
    torch.nn.init.uniform_(P12.weight, -0.1, 0.1)
    torch.nn.init.uniform_(P21.weight, -0.1, 0.1)


    # # torch.nn.init.orthogonal_(P12.weight)
    # # torch.nn.init.orthogonal_(P21.weight)

    logger.info(f"P12 dimension: {P12.weight.shape}, P21 dimension: {P21.weight.shape}")

    P12.to(device)
    P21.to(device)

    optimizer_model_1 = torch.optim.Adam(model_1.parameters(), lr=args.lr)
    optimizer_model_2 = torch.optim.Adam(model_2.parameters(), lr=args.lr)

    # restriction_updater = restriction_maps_optimizer(
    #     P12, 
    #     P21, 
    #     lr= args.lr_restriction_maps, 
    #     lambda_reg= args.lambda_reg
    #     ) 
    
    restriction_updater_P12 = restriction_maps_optimizer(
        P12, 
        P21, 
        lr= args.lr_restriction_maps, 
        lambda_reg= args.lambda_reg
        ) 
    
    restriction_updater_P21 = restriction_maps_optimizer(
        P12, 
        P21, 
        lr= args.lr_restriction_maps, 
        lambda_reg= args.lambda_reg
        ) 


    logger.info(f"Restrictions Maps: {restriction_updater_P12}, {restriction_updater_P21}")

    loss_1 = nn.CrossEntropyLoss()
    loss_2 = nn.CrossEntropyLoss()

    


    logger.info(f"Create Tensorboard Writer will be saved to {args.output_dir}/runs/sheaf_fmtl_experiment")
    writer = SummaryWriter(log_dir="runs/sheaf_fmtl_experiment")


    # save best model weights
    best_loss_model1 = float("inf")
    best_loss_model2 = float("inf")
    wait1, wait2 = 0, 0

    stop_model1 = False
    stop_model2 = False



    logger.info(f"Start from {args.start_epoch} to {args.epochs} epochs")
    start_time = time.time()


    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train_1.sampler.set_epoch(epoch)

        model_1.train()
        model_2.train()
        total_loss_model_1 = 0.0
        total_loss_model_2 = 0.0

        it1 = iter(data_loader_train_1)
        it2 = iter(data_loader_train_2)

        i = 0
        while True:
            try:
                samples1, labels1 = next(it1)
            except StopIteration:
                samples1 = None
                labels1 = None

            try:
                samples2, labels2 = next(it2)
            except StopIteration:
                samples2 = None
                labels2 = None

            if samples1 is None and samples2 is None:
                break
        
            if samples1 is not None and not stop_model1:
                samples1 = samples1.to(device, non_blocking=True)
              
                labels1 = labels1.to(device, non_blocking=True)

                optimizer_model_1.zero_grad()
                outputs_1 = model_1(samples1)
                loss_model_1 = loss_1(outputs_1, labels1)
                loss_model_1.backward()
            
                # Clip gradients for model 1
                torch.nn.utils.clip_grad_norm_(model_1.parameters(), max_norm=1.0)
                optimizer_model_1.step()

                total_loss_model_1 += loss_model_1.item()

            if samples2 is not None and not stop_model2:
                samples2 = samples2.to(device, non_blocking=True)
                labels2 = labels2.to(device, non_blocking=True)
                
                optimizer_model_2.zero_grad()
                outputs_2 = model_2(samples2)
                loss_model_2 = loss_2(outputs_2, labels2)
                loss_model_2.backward()

                # Clip gradients for model 2
                torch.nn.utils.clip_grad_norm_(model_2.parameters(), max_norm=1.0)
                optimizer_model_2.step()

                total_loss_model_2 += loss_model_2.item()

            # Apply update directly to the model weights
            with torch.no_grad():
                # # Theta vectors before update
                theta1 = model_1.fc1.weight
                theta2 = model_2.fc1.weight

                # Compute your custom update
                # For example: theta1_new = theta1 - alpha * (gradient + lambda * (P12.T @ (P12 @ theta1 - P21 @ theta2)))
                # Make sure gradients are included if you want autograd tracking
                if not stop_model1:
                    theta1 -= args.lr * (theta1.grad + args.lambda_reg *  (P12(theta1) - P21(theta2))@ P12.weight)
                    theta1.grad.zero_() #BORRAR TEST
                if not stop_model2:
                    theta2 -= args.lr * (theta2.grad + args.lambda_reg *  (P21(theta2) - P12(theta1))@ P21.weight)
                    theta2.grad.zero_()  #BORRAR TEST

            theta1 = model_1.fc1.weight.detach()
            theta2 = model_2.fc1.weight.detach()


            # # update P12 and P21 in the direction of reducing discrepancy
            if not stop_model1:
                restriction_updater_P12.step(theta1, theta2)
            if not stop_model2:
                restriction_updater_P21.step(theta1, theta2)

            # restriction_updater.step(theta1, theta2) 

            # Log training losses (averaged per epoch)
            avg_train_loss_model1 = total_loss_model_1 / len(data_loader_train_1)
            avg_train_loss_model2 = total_loss_model_2 / len(data_loader_train_2)

            writer.add_scalar("Train/Loss_Model1", avg_train_loss_model1, epoch)
            writer.add_scalar("Train/Loss_Model2", avg_train_loss_model2, epoch)

            if i % 100 == 99:  # print every 100 mini-batches
                
                if samples1 is not None:
                    logger.info(f"[Model 1: Epoch {epoch+1}, Batch {i+1}] loss: {total_loss_model_1/100}")
                    total_loss_model_1 = 0.0
                if samples2 is not None:
                    logger.info(f"[Model 2: Epoch {epoch+1}, Batch {i+1}] loss: {total_loss_model_2/100}")
                    total_loss_model_2 = 0.0


            i += 1


        # ----- Validation phase -----
        logger.info(f"-------VALIDATION Epoch: {epoch}")

        model_1.eval()
        model_2.eval()

        # EXTRACT DATA FROM MODELS 
    
        model_1_projection_list = []
        model_2_projection_list = []

        def hook_model1(module, input, output):
            # output here is fc1(x) (pre-ReLU), so apply ReLU manually
            flattened = output.view(output.size(0), -1)
            model_1_projection_list.append(flattened.detach().cpu())
            

        def hook_model2(module, input, output):
            # output here is fc1(x) (pre-ReLU), so apply ReLU manually
            flattened = output.view(output.size(0), -1)
            model_2_projection_list.append(flattened.detach().cpu())

        # Register the hook
        hook1 = model_1.conv_block3.register_forward_hook(hook_model1)
        hook2 = model_2.conv_block3.register_forward_hook(hook_model2)




        correct_1, total_1, test_loss_1 = 0, 0, 0.0
        correct_2, total_2, test_loss_2 = 0, 0, 0.0

        with torch.no_grad():
            for i, (samples, labels) in enumerate(data_loader_test_1):
                samples = samples.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                outputs_1 = model_1(samples)   
                loss_model_1 = loss_1(outputs_1, labels)
                test_loss_1 += loss_model_1.item()

                preds = outputs_1.argmax(dim=1)
                correct_1 += (preds == labels).sum().item()
                total_1 += labels.size(0)

            acc_1 = correct_1 / total_1
            logger.info(f"Model 1 - Loss: {test_loss_1/len(data_loader_test_1):.4f}, Acc: {acc_1:.4f}")


            for i, (samples, labels) in enumerate(data_loader_test_2):
                samples = samples.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                outputs_2 = model_2(samples)   
                loss_model_2 = loss_2(outputs_2, labels)
                test_loss_2 += loss_model_2.item()

                preds = outputs_2.argmax(dim=1)
                correct_2 += (preds == labels).sum().item()
                total_2 += labels.size(0)

            acc_2 = correct_2 / total_2
            logger.info(f"Model 2 - Loss: {test_loss_2/len(data_loader_test_2):.4f}, Acc: {acc_2:.4f}")


            # Remove hooks after evaluation
            hook1.remove()
            hook2.remove()

            # ---- Verify discrepancy between restriction maps ----
            # Concatenate all batches: [num_samples, fc1_dim]
            projection_all_model_1 = torch.cat(model_1_projection_list, dim=0).to(device) 
            projection_all_model_2 = torch.cat(model_2_projection_list, dim=0).to(device)  

            # Project with P12
            proj_all_model1_P12 = P12(projection_all_model_1).detach().cpu()
            # Project with P21
            proj_all_model2_P21 = P21(projection_all_model_2).detach().cpu()
            discrepancy = proj_all_model1_P12 - proj_all_model2_P21

            normalized_discrepancy = torch.sum(discrepancy**2) / (projection_all_model_1.size(0) * discrepancy.size(1))


          
            logger.info(f"Restriction map discrepancy: {discrepancy.mean().item()} Discrepancy Similarity: {  torch.sum(discrepancy**2).item()}, normalized_discrepancy: {  normalized_discrepancy.item()}")

            avg_test_loss_model1 = test_loss_1 / len(data_loader_test_1)
            avg_test_loss_model2 = test_loss_2 / len(data_loader_test_2)

            # Log validation metrics
            writer.add_scalar("Test/Loss_Model1", avg_test_loss_model1, epoch)
            writer.add_scalar("Test/Loss_Model2", avg_test_loss_model2, epoch)
            writer.add_scalar("Test/Acc_Model1", acc_1, epoch)
            writer.add_scalar("Test/Acc_Model2", acc_2, epoch)

            # Also log discrepancy between projections
            writer.add_scalar("RestrictionMaps/MSE_Discrepancy", discrepancy.mean().item(), epoch)
            writer.add_scalar("RestrictionMaps/CosineSim", torch.sum(discrepancy**2).item(), epoch)

            # ----- Model 1 -----
            if not stop_model1:
                if avg_test_loss_model1 < best_loss_model1:
                    best_loss_model1 = avg_test_loss_model1
                    wait1 = 0
                    save_checkpoint(model_1, "best_model1.pth", extra={"P12": P12.state_dict()})
                else:
                    wait1 += 1
                    logger.info(f"Model 1 no improvement {wait1}/{args.patience}")
                    if wait1 >= args.patience:
                        logger.info("Stopping Model 1 training (patience reached).")
                        stop_model1 = True
            else:
                logger.info("Load best model 1")
                checkpoint = torch.load("./saved_model/best_model1.pth")
                model_1.load_state_dict(checkpoint["model"])
                P12.load_state_dict(checkpoint["P12"])

            # ----- Model 2 -----
            if not stop_model2:
                if avg_test_loss_model2 < best_loss_model2:
                    best_loss_model2 = avg_test_loss_model2
                    wait2 = 0
                    save_checkpoint(model_2, "best_model2.pth", extra={"P21": P21.state_dict()})
                else:
                    wait2 += 1
                    logger.info(f"Model 2 no improvement {wait2}/{args.patience}")
                    if wait2 >= args.patience:
                        logger.info("Stopping Model 2 training (patience reached).")
                        stop_model2 = True
            else:
                logger.info("Load best model 2")
                checkpoint = torch.load("./saved_model/best_model2.pth")
                model_2.load_state_dict(checkpoint["model"])
                P21.load_state_dict(checkpoint["P21"])


            if stop_model1 and stop_model2:
                logger.info(" Both models stopped. Ending training loop.")
                break

    writer.close()

            

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f"Training time {total_time_str}")



if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)


