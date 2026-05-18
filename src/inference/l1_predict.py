import torch
import numpy as np
import pandas as pd
from src.models.cnn1d import CNN1D
from src.utils.paths import get_data_path
from src.utils.logging import setup_logger
from src.config.loader import load_config

logger = setup_logger('L1_Predictor', 'logs/l1_predict.log')

def load_model(config):
    model = CNN1D(num_classes=config['model']['parameters']['num_classes'])
    model.load_state_dict(torch.load(config['model']['weights_path']))
    model.eval()
    return model

def predict(ecg_segment, model):
    with torch.no_grad():
        ecg_tensor = torch.tensor(ecg_segment, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        prediction = model(ecg_tensor)
        predicted_class = torch.argmax(prediction, dim=1).item()
    return predicted_class

def main():
    config = load_config('configs/l1_beat_classifier.yaml')
    model = load_model(config)

    # Load ECG segments for prediction
    data_path = get_data_path()
    ecg_data = pd.read_csv(f'{data_path}/ecg_segments.csv')  # Assuming segments are stored in a CSV file

    predictions = []
    for index, row in ecg_data.iterrows():
        ecg_segment = row.values  # Extract ECG segment
        predicted_class = predict(ecg_segment, model)
        predictions.append(predicted_class)

    ecg_data['predicted_class'] = predictions
    ecg_data.to_csv(f'{data_path}/predictions.csv', index=False)
    logger.info("Predictions saved to predictions.csv")

if __name__ == "__main__":
    main()