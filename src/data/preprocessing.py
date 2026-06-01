import numpy as np
import wfdb
from scipy.signal import resample_poly
from math import gcd
import os

# Window definition (samples) for beat segmentation centered on R-peak
PRE_SAMPLES = 100
POST_SAMPLES = 150
WINDOW_SIZE = PRE_SAMPLES + POST_SAMPLES

def resample_signal(signal: np.ndarray, 
                    fs_orig: int, 
                    fs_target: int = 360) -> np.ndarray:
    """
    Resample signal from fs_orig to fs_target Hz.
    Uses polyphase filtering (resample_poly) — better than resample()
    for ECG because preserves morphology without ringing.
    
    Example: INCART 257Hz → MIT-BIH 360Hz
    """
    if fs_orig == fs_target:
        return signal
    g   = gcd(fs_orig, fs_target)
    up  = fs_target // g   # 360 // gcd(257,360)
    dn  = fs_orig   // g   # 257 // gcd(257,360)
    return resample_poly(signal, up, dn)

def apply_bandpass(ecg_signal: np.ndarray, fs: int = 360, low_hz: float = 0.5, high_hz: float = 40.0, order: int = 2) -> np.ndarray:
    """Bandpass filter for ECG preprocessing (baseline wander + high-frequency noise).

    Default band (0.5-40 Hz) preserves P-QRS-T morphology while removing drift.
    This is used as step (1) in the end-to-end dataset builder.
    """

    from scipy.signal import butter, filtfilt

    x = np.asarray(ecg_signal, dtype=float).reshape(-1)
    nyq = 0.5 * float(fs)
    low = float(low_hz) / nyq
    high = float(high_hz) / nyq

    if not (0 < low < high < 1):
        raise ValueError(f"Invalid bandpass: low_hz={low_hz}, high_hz={high_hz}, fs={fs}")

    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, x)


def load_ecg_data(record_name, sampling_rate=360):
    record = wfdb.rdrecord(record_name, sampto=250)
    ecg_signal = record.p_signal[:, 0]  # Assuming the first channel is the ECG signal
    return ecg_signal


def preprocess_ecg(ecg_signal):
    # Normalize the ECG signal
    ecg_signal = (ecg_signal - np.mean(ecg_signal)) / np.std(ecg_signal)
    return ecg_signal


def segment_ecg(ecg_signal, r_peaks, pre_samples: int = PRE_SAMPLES, post_samples: int = POST_SAMPLES):
    """Segment ECG beats around R-peaks using an asymmetric window.

    Returns segments of fixed length pre_samples+post_samples.
    """

    segments = []
    win_len = pre_samples + post_samples

    for r_peak in r_peaks:
        start = int(r_peak) - pre_samples
        end = int(r_peak) + post_samples
        if start >= 0 and end <= len(ecg_signal) and (end - start) == win_len:
            segments.append(ecg_signal[start:end])

    return np.array(segments)


def save_preprocessed_data(segments, save_path):
    np.save(save_path, segments)


def load_and_preprocess(record_name, r_peaks, save_path):
    ecg_signal = load_ecg_data(record_name)
    preprocessed_signal = preprocess_ecg(ecg_signal)
    segments = segment_ecg(preprocessed_signal, r_peaks)
    save_preprocessed_data(segments, save_path)