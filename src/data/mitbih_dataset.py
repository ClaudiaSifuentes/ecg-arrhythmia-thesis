import wfdb
import numpy as np

from .aami_mapping import map_symbol_to_aami_class

# Window definition (samples) for beat segmentation centered on R-peak
PRE_SAMPLES = 100
POST_SAMPLES = 150
WINDOW_SIZE = PRE_SAMPLES + POST_SAMPLES


class MITBIHDataset:
    def __init__(self, record_name, sampling_rate=360):
        self.record_name = record_name
        self.sampling_rate = sampling_rate
        self.ecg_data = None
        self.labels = None
        self.r_peaks = None

    def load_data(self):
        # NOTE: do not truncate the record to 250 samples; we need full-length signal
        record = wfdb.rdrecord(self.record_name)
        self.ecg_data = record.p_signal[:, 0]  # single lead

        ann = wfdb.rdann(self.record_name, "atr")
        self.labels = ann.symbol
        self.r_peaks = ann.sample

    def preprocess_data(self):
        # Normalize ECG data (z-score)
        self.ecg_data = (self.ecg_data - np.mean(self.ecg_data)) / np.std(self.ecg_data)

    def get_ecg_segments_and_labels(
        self,
        pre_samples: int = PRE_SAMPLES,
        post_samples: int = POST_SAMPLES,
    ):
        """Return (segments, y) filtered to mapped AAMI classes.

        - Uses asymmetric beat window: [R-pre_samples, R+post_samples)
        - Drops symbols that map to None (unknown/non-beat)
        - Ensures len(segments) == len(y)
        """

        segments = []
        y = []
        win_len = pre_samples + post_samples

        for r_peak, sym in zip(self.r_peaks, self.labels):
            cls = map_symbol_to_aami_class(str(sym))
            if cls is None:
                continue

            start = int(r_peak) - pre_samples
            end = int(r_peak) + post_samples
            if start < 0 or end > len(self.ecg_data) or (end - start) != win_len:
                continue

            segments.append(self.ecg_data[start:end])
            y.append(int(cls))

        segments_arr = np.asarray(segments, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.int64)

        if len(segments_arr) != len(y_arr):
            raise RuntimeError(
                f"Segment/label alignment error: len(segments)={len(segments_arr)} != len(y)={len(y_arr)}"
            )

        return segments_arr, y_arr

    def get_data(self):
        self.load_data()
        self.preprocess_data()
        return self.get_ecg_segments_and_labels()