import os
import yaml
import torch
from src.utils.logging import setup_logger
from src.utils.reproducibility import set_global_seeds
from src.data.mitbih_dataset import MITBIHDataset
from src.models.cnn1d import CNN1D
from src.training.trainer import Trainer

# Load configuration
def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main():
    # Set global seed for reproducibility
    set_global_seeds(42)

    # Setup logger
    logger = setup_logger('train_l1', os.path.join('logs', 'train_l1.log'))

    # Load configuration
    config = load_config('configs/l1_beat_classifier.yaml')
    logger.info("Loaded configuration: %s", config)

    # Initialize dataset
    dataset = MITBIHDataset(config['data'])
    logger.info("Dataset initialized with %d samples", len(dataset))

    # Initialize model
    model = CNN1D(config['model'])
    logger.info("Model initialized: %s", model)

    # Initialize trainer
    trainer = Trainer(model, dataset, config['training'])
    logger.info("Trainer initialized")

    # Start training
    trainer.train()
    logger.info("Training completed")

if __name__ == "__main__":
    main()