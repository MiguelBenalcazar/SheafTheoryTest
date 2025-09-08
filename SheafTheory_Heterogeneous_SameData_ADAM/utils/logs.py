from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from pathlib import Path
import logging, sys
import json

def setup_run(args):
    # Create output_dir/runs/<timestamp>
    run_dir = Path(args.output_dir) / "runs" / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    # Log file path
    log_file = run_dir / "training.log"

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, mode="w"),
        ],
    )
    logger = logging.getLogger(__name__)

    # TensorBoard writer
    writer = SummaryWriter(log_dir=run_dir)

    # Save training arguments
    args_filepath = run_dir / "args.json"
    with open(args_filepath, "w") as f:
        json.dump(vars(args), f, indent=4)
    logger.info(f"Saved training arguments to: {args_filepath}")

    logger.info(f"Logs will be saved in: {log_file}")
    logger.info(f"TensorBoard data will be saved in: {run_dir}")

    return logger, writer, run_dir
