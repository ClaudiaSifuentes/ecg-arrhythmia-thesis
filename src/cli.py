import argparse
import logging
from src.utils.logging import setup_logger
from src.config.loader import load_config
from src.training.trainer import Trainer

def main():
    # Set up logging
    logger = setup_logger('ECG_Arrhythmia_Detection', 'logs/project.log')

    # Argument parser
    parser = argparse.ArgumentParser(description='ECG Arrhythmia Detection CLI')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to the configuration file')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--score', action='store_true', help='Score the model')
    parser.add_argument('--detect', action='store_true', help='Detect events using the model')

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    if args.train:
        logger.info("Starting training process...")
        trainer = Trainer(config)
        trainer.train()
    elif args.score:
        logger.info("Starting scoring process...")
        # Call scoring function here
    elif args.detect:
        logger.info("Starting event detection process...")
        # Call detection function here
    else:
        logger.error("No action specified. Use --train, --score, or --detect.")

if __name__ == '__main__':
    main()