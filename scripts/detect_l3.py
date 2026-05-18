import os
import yaml
from src.utils.logging import setup_logger
from src.utils.reproducibility import set_global_seeds
from src.utils.paths import get_configs_path
from src.inference.l2_score import compute_l2_scores
from src.inference.l3_events import detect_events

# Set up logging
logger = setup_logger('detect_l3', os.path.join('logs', 'detect_l3.log'))

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main():
    # Set global seed for reproducibility
    set_global_seeds(42)

    # Load configuration for L3 event detection
    config_path = os.path.join(get_configs_path(), 'l3_event_detection.yaml')
    config = load_config(config_path)

    # Compute L2 scores
    l2_scores = compute_l2_scores(config)

    # Detect clinical events based on L2 scores
    events = detect_events(l2_scores, config)

    # Log detected events
    logger.info(f'Detected events: {events}')

if __name__ == "__main__":
    main()