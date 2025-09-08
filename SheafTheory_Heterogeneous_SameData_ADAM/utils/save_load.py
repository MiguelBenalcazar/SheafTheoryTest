import torch
import os
import glob
import logging
import json

logger = logging.getLogger(__name__)

# def save_checkpoint(model, path, extra=None):
#     checkpoint = {"model": model.state_dict()}
#     if extra is not None:
#         checkpoint.update(extra)
#     torch.save(checkpoint, f"./saved_model/{path}")
#     logger.info(f"Saved checkpoint at ./saved_model/{path}")


# def save_checkpoint(model, 
#                     batch:int, 
#                     acc:float,
#                     prefix="model", 
#                     extra=None, 
#                     keep_last=2):
    
#     os.makedirs("./saved_model", exist_ok=True)

#     # Format filename: e.g. model_batch00010_acc0.8123.pth
#     filename = f"{prefix}_batch{batch:05d}_acc{acc:.4f}.pth"
#     path = os.path.join("./saved_model", filename)

#     checkpoint = {"model": model.state_dict(), "batch": batch, "acc": acc}
#     if extra is not None:
#         checkpoint.update(extra)

#     torch.save(checkpoint, path)
#     logger.info(f"Saved checkpoint at {path}")

#     # --- Keep only last N checkpoints ---
#     ckpts = sorted(glob.glob("./saved_model/*.pth"), key=os.path.getmtime, reverse=True)
#     for old_ckpt in ckpts[keep_last:]:
#         try:
#             os.remove(old_ckpt)
#             logger.info(f"Deleted old checkpoint: {old_ckpt}")
#         except Exception as e:
#             logger.warning(f"Could not delete {old_ckpt}: {e}")



def save_checkpoint(model, 
                    batch:int, 
                    monitor_value:float,
                    monitor:str="acc",   # "acc" or "loss"
                    prefix="model", 
                    extra=None, 
                    keep_last=2,
                    is_best=False):
    """
    Save a checkpoint with metric (loss or acc) in filename and dict.

    Args:
        model: torch.nn.Module
        batch: int, epoch or step number
        monitor_value: float, value of the monitored metric
        monitor: str, either "acc" or "loss"
        prefix: str, prefix for filename
        extra: dict, extra state_dicts (e.g. restriction maps)
        keep_last: int, number of rolling checkpoints to keep
        is_best: bool, whether this is the best checkpoint (also saves best_*.pth)
    """
    os.makedirs("./saved_model", exist_ok=True)

    # Format filename dynamically depending on monitor
    filename = f"{prefix}_batch_{batch}_{monitor}_{monitor_value:.4f}.pth"
    path = os.path.join("./saved_model", filename)

    # Prepare checkpoint
    checkpoint = {"model": model.state_dict(), "batch": batch, monitor: monitor_value}
    if extra is not None:
        checkpoint.update(extra)

    # Save checkpoint
    torch.save(checkpoint, path)
    logger.info(f"Saved checkpoint at {path}")

    # Save best separately
    if is_best:
        best_path = os.path.join("./saved_model", f"best_{prefix}.pth")
        torch.save(checkpoint, best_path)
        logger.info(f"Updated BEST checkpoint at {best_path}")

    

    # --- best checkpoint + metadata ---
    if is_best:
        best_path = os.path.join("./saved_model", f"best_{prefix}.pth")
        torch.save(checkpoint, best_path)
        logger.info(f"Updated BEST checkpoint at {best_path}")

    # --- Keep only last N rolling checkpoints ---
    ckpts = sorted(glob.glob(f"./saved_model/{prefix}_batch*.pth"), key=os.path.getmtime, reverse=True)
    for old_ckpt in ckpts[keep_last:]:
        try:
            os.remove(old_ckpt)
            logger.info(f"Deleted old checkpoint: {old_ckpt}")
        except Exception as e:
            logger.warning(f"Could not delete {old_ckpt}: {e}")




def load_checkpoint(path, model, restriction_map, extra=['model', 'P12']):
    if not os.path.isfile(path):
        logger.error(f"Checkpoint file not found: {path}")
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    try:
        ckpt1 = torch.load(path)  # safer across devices
        model.load_state_dict(ckpt1[extra[0]])
        restriction_map.load_state_dict(ckpt1[extra[1]])
        logger.info(f"Checkpoint loaded successfully for {path}.")
        

    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}")
        raise RuntimeError(f"Error loading checkpoint: {e}")
    




# def load_checkpoint(path, model, restriction_map, extra=['model', 'P12']):
#     # Load best Model 1 and P12
#     ckpt1 = torch.load(path)  # safer if loading across devices
#     try:
#         model.load_state_dict(ckpt1[extra[0]])
#         restriction_map.load_state_dict(ckpt1[extra[1]])
#         logger.info("Checkpoint loaded successfully.")

#     except Exception as e:
#         logger.error(f"Error loading checkpoint: {e}")
#         raise f"Error loading checkpoint: {e}"

    

