import os
import sys
import logging
from src.utils.reproducibility import set_global_seeds
from src.utils.paths import get_data_path
from src.config.loader import load_config
from src.data.mitbih_dataset import MITBIHDataset
from src.training.trainer import Trainer

def main():
    # Set global seeds for reproducibility
    set_global_seeds(42)

    # Load configuration
    config = load_config('configs/default.yaml')

    # Initialize logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Load dataset
    data_path = get_data_path()
    dataset = MITBIHDataset(data_path, config['data'])

    # Initialize trainer
    trainer = Trainer(config['training'], dataset)

    # Start training
    logger.info("Starting training...")
    trainer.train()

if __name__ == "__main__":
    main()