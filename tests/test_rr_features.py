import pytest
from src.data.rr_features import extract_rr_features

def test_extract_rr_features():
    # Sample ECG signal and R-peak indices for testing
    ecg_signal = [0.1, 0.2, 0.3, 0.4, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0] * 25  # Simulated ECG signal
    r_peaks = [4, 29]  # Indices of R-peaks in the simulated ECG signal

    # Expected RR features
    expected_features = {
        'RR_actual': 0.1,
        'RR_prev': 0.1,
        'RR_mean': 0.1,
        'SDNN': 0.0,
        'RMSSD': 0.0,
        'RR_ratio': 1.0
    }

    # Extract RR features
    features = extract_rr_features(ecg_signal, r_peaks)

    # Assert that the extracted features match the expected features
    for key in expected_features:
        assert features[key] == expected_features[key], f"Expected {key} to be {expected_features[key]}, but got {features[key]}"

if __name__ == "__main__":
    pytest.main()