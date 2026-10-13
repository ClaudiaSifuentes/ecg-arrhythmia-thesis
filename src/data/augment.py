"""Train-time beat augmentation (applied per batch, only on train).

Config examples:
    augment: {p: 0.5, gauss_std: 0.05, baseline_amp: 0.1, shift_max: 5, scale: 0.1}
    augment: {p: 0.5, nstdb_snr: [6, 24]}   # real NSTDB noise (training half only)
"""

from __future__ import annotations

import math

import numpy as np
import torch


def make_augmenter(cfg: dict | None, seed: int):
    if not cfg:
        return None
    g = torch.Generator().manual_seed(seed + 7)
    p = float(cfg.get("p", 0.5))
    gauss = float(cfg.get("gauss_std", 0.0))
    base_amp = float(cfg.get("baseline_amp", 0.0))
    shift = int(cfg.get("shift_max", 0))
    scale = float(cfg.get("scale", 0.0))

    nstdb_snr = cfg.get("nstdb_snr")  # e.g. [6, 24]: REAL noise, first half of NSTDB records
    if nstdb_snr:
        from src.data.nstdb_noise import NOISE_TYPES, load_nstdb_noise

        noise = load_nstdb_noise(part="train")
        types = list(cfg.get("nstdb_types", NOISE_TYPES))
        np_rng = np.random.default_rng(seed + 11)

    def aug(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        B, L = x.shape
        m = (torch.rand(B, 1, generator=g) < p).float()
        out = x
        if nstdb_snr:
            sel = np.nonzero(m[:, 0].numpy() > 0)[0]
            add = np.zeros((B, L), dtype=np.float32)
            xs = out.numpy()
            for i in sel:
                src = noise[types[np_rng.integers(len(types))]]
                start = int(np_rng.integers(0, len(src) - L))
                seg = src[start:start + L] - src[start:start + L].mean()
                snr = np_rng.uniform(nstdb_snr[0], nstdb_snr[1])
                gain = np.sqrt((xs[i].var() + 1e-8) / ((seg.var() + 1e-12) * 10 ** (snr / 10)))
                add[i] = gain * seg
            out = out + torch.from_numpy(add)
        if shift > 0:
            s = torch.randint(-shift, shift + 1, (B,), generator=g)
            ar = (torch.arange(L).unsqueeze(0) - s.unsqueeze(1)).clamp(0, L - 1)
            out = torch.where(m.bool(), torch.gather(out, 1, ar), out)
        if scale > 0:
            out = out * (1 + m * (torch.rand(B, 1, generator=g) * 2 - 1) * scale)
        if base_amp > 0:
            # low-frequency drift (0.1-0.5 Hz over a 0.69 s window ~ slow ramp/curve)
            f = torch.rand(B, 1, generator=g) * 0.4 + 0.1
            ph = torch.rand(B, 1, generator=g) * 2 * math.pi
            t = torch.arange(L).float().unsqueeze(0) / 360.0
            out = out + m * base_amp * torch.sin(2 * math.pi * f * t + ph)
        if gauss > 0:
            out = out + m * gauss * torch.randn(B, L, generator=g)
        return out

    return aug
