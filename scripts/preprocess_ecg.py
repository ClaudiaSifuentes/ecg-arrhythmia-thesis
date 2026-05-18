import os
import wfdb
import numpy as np
from src.data.preprocessing import preprocess_ecg_signal
from src.data.rr_features import extract_rr_features
from src.utils.paths import get_data_path

def load_ecg_data(record_name):
    record = wfdb.rdrecord(record_name)
    return record.p_signal.flatten()

def preprocess_ecg(record_name):
    ecg_signal = load_ecg_data(record_name)
    preprocessed_signal = preprocess_ecg_signal(ecg_signal)
    return preprocessed_signal

def extract_features(ecg_signal):
    rr_features = extract_rr_features(ecg_signal)
    return rr_features

def main():
    # Define the record name (MIT-BIH dataset example)
    record_name = os.path.join(get_data_path(), 'mitbih', '100')  # Adjust path as necessary

    # Preprocess ECG data
    preprocessed_signal = preprocess_ecg(record_name)

    # Extract RR features
    features = extract_features(preprocessed_signal)

    # Save or return the processed data and features as needed
    print("Preprocessed Signal:", preprocessed_signal)
    print("Extracted Features:", features)

if __name__ == "__main__":
    main()