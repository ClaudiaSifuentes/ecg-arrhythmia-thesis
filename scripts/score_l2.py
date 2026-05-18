import os
import yaml
import numpy as np
from src.utils.logging import setup_logger
from src.utils.reproducibility import set_global_seeds
from src.data.mitbih_dataset import load_mitbih_data
from src.inference.l1_predict import predict_l1
from src.inference.l2_score import compute_l2_scores

# Setup logger
logger = setup_logger('score_l2', os.path.join('logs', 'score_l2.log'))

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main():
    # Load configuration
    config = load_config('configs/l2_temporal_scoring.yaml')
    
    # Set global seed for reproducibility
    set_global_seeds(config['seed'])

    # Load MIT-BIH dataset
    logger.info("Loading MIT-BIH dataset...")
    data = load_mitbih_data()

    # Predict L1 scores
    logger.info("Predicting L1 scores...")
    l1_predictions = predict_l1(data)

    # Compute L2 anomaly scores
    logger.info("Computing L2 anomaly scores...")
    l2_scores = compute_l2_scores(l1_predictions, config['evaluation']['threshold'])

    # Save or process L2 scores as needed
    logger.info("L2 anomaly scores computed successfully.")

if __name__ == "__main__":
    main()