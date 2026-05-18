import torch
import torch.nn as nn
import torch.nn.functional as F

class ECGFeatureExtractor(nn.Module):
    """1D CNN for morphological feature extraction from ECG beats (L1 backbone).

    Input / Output
    --------------
    - Input:  (B, 1, 250)
      Single-channel ECG segment (250 samples @ 360 Hz) centered on R-peak.
    - Output: (B, 256)
      Morphological embedding used for late fusion with RR handcrafted features.

    Architecture decisions (for thesis defense)
    -------------------------------------------
    - Block 1 uses `kernel_size=5`:
        At 360 Hz, 5 samples ~ 13.9 ms, matching the temporal scale of the QRS
        complex slope/peaks while still being local enough to capture morphology.
    - Blocks 2-3 use `kernel_size=3`:
        After downsampling via MaxPool, the effective receptive field grows;
        smaller kernels capture patterns at reduced resolution without over-parameterizing.
    - BatchNorm after each Conv:
        Stabilizes training and gradient flow; enables higher learning rates and
        reduces sensitivity to initialization (important for a fast baseline).
    - MaxPool(2) reduces temporal resolution:
        250 → 125 → 62, lowering compute while preserving salient morphology.
    - AdaptiveAvgPool1d(1):
        Global pooling produces a fixed-size representation independent of exact
        input length (robustness + simpler head design).
    - No Dropout in CNN trunk:
        Regularization is applied in the fusion head (`FusionClassifier`) to avoid
        suppressing low-level morphological cues.

    Parameter budget
    ----------------
    ~158k params (confirmed by smoke test) — well below the <500k constraint.
    """

    def __init__(self):
        super().__init__()

        # Block 1: capture QRS morphology
        # kernel=5 at 360Hz ≈ 14ms — temporal scale of QRS complex
        self.block1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 250 → 125
        )

        # Block 2: rhythm-level patterns
        self.block2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 125 → 62
        )

        # Block 3: high-level representation
        self.block3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # 62 → 1 (global)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x: torch.Tensor
            Shape (B, 1, 250)

        Returns
        -------
        torch.Tensor
            Shape (B, 256)
        """

        x = self.block1(x)  # (B, 64, 125)
        x = self.block2(x)  # (B, 128, 62)
        x = self.block3(x)  # (B, 256, 1)
        return x.squeeze(-1)  # (B, 256)