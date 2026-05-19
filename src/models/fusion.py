import torch
import torch.nn as nn

from src.models.cnn1d import ECGFeatureExtractor


class FusionClassifier(nn.Module):
    """Hybrid CNN+RR classifier with late fusion.

    Input / Output
    --------------
    - Input:
        ecg: (B, 1, 250)
        rr:  (B, 6)
    - Output:
        logits: (B, 3) raw class scores for [N, SVEB, VEB]

    Architecture decisions (for thesis defense)
    -------------------------------------------
    - Late fusion:
        CNN and RR are processed independently, then concatenated. This separates
        morphological information (CNN) from temporal dynamics (RR features).
        It also enables ablations:
          * CNN-only: set rr input to zeros.
          * RR-only: replace CNN embedding with zeros / bypass CNN.
    - Concatenation [h_cnn(256) | rr(6)] -> z(262):
        Simplest valid fusion that preserves interpretability of RR features.
        RR is not projected before fusion to avoid transforming clinically
        meaningful handcrafted variables.
    - Dense(262->128) + ReLU:
        Learns a non-linear interaction between morphology and RR features.
    - Dropout(0.5) in the head (not in CNN trunk):
        Regularizes the classifier while keeping morphological extraction stable.
    - Dense(128->3):
        Final logits without softmax (PyTorch CrossEntropyLoss expects logits).

    Parameter budget
    ----------------
    CNN backbone: 124,544 params (from previous smoke test)
    Head: ~33k params
    Total: ~158k params, safely under the 500k constraint.
    """

    def __init__(self, dropout: float = 0.5, n_classes: int = 3, use_rr: bool = True):
        super().__init__()
        self.use_rr = use_rr
        self.cnn = ECGFeatureExtractor()

        self.head = nn.Sequential(
            nn.Linear(256 + 6, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, ecg: torch.Tensor, rr: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            ecg: (B, 1, 250) ECG segment tensor
            rr:  (B, 6) RR interval handcrafted features

        Returns:
            logits: (B, 3) raw class scores (no softmax)
        """

        h = self.cnn(ecg)  # (B, 256)
        if not self.use_rr:
            rr = torch.zeros_like(rr)
        z = torch.cat([h, rr], dim=1)  # (B, 262)
        return self.head(z)  # (B, 3)