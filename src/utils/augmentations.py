"""
State-of-the-art waveform + spectrogram augmentations
for Speech Commands / general keyword-spotting.
Author: <you> – 2025-04-26
"""

from __future__ import annotations
import random, math, pathlib
from typing import Sequence, Callable, Optional

import torch
import torchaudio
import torchaudio.functional as F
from torchaudio.transforms import FrequencyMasking, TimeMasking

SR = 16_000         # sample rate (Hz)
FIXED_LEN = 16_000  # 1-second clips

# --------------------------------------------------------------------------- #
#                              helper utilities                               #
# --------------------------------------------------------------------------- #

def pad_or_trim(wav: torch.Tensor, length: int = FIXED_LEN) -> torch.Tensor:
    """Center-crop / pad a waveform to exactly `length` samples."""
    L = wav.shape[-1]
    if L == length:
        return wav
    if L > length:
        start = (L - length) // 2
        return wav[..., start : start + length]
    # pad
    pad_left = (length - L) // 2
    pad_right = length - L - pad_left
    return torch.nn.functional.pad(wav, (pad_left, pad_right))

def snr_mix(clean: torch.Tensor, noise: torch.Tensor, snr_db: float) -> torch.Tensor:
    """Mix `noise` into `clean` at a target SNR (dB)."""
    noise = pad_or_trim(noise, clean.shape[-1])
    clean_power = clean.pow(2).mean()
    noise_power = noise.pow(2).mean().clamp(min=1e-9)
    factor = (clean_power / (10 ** (snr_db / 10) * noise_power)).sqrt()
    return clean + factor * noise

# --------------------------------------------------------------------------- #
#                            waveform-domain aug                              #
# --------------------------------------------------------------------------- #

class WaveformAug:
    """
    Waveform-level augmentation callable.
    Pass an UrbanSound8K / WHAM! dataset (or list of tensors) as `bg_bank`.
    """

    def __init__(
        self,
        bg_bank: Sequence[torch.Tensor],
        rir_bank: Optional[Sequence[torch.Tensor]] = None,
        max_shift_ms: int = 200,
    ):
        self.bg_bank = bg_bank
        self.rir_bank = rir_bank or []
        self.max_shift = int(max_shift_ms * SR / 1_000)

    # ----- individual primitive ops -------------------------------------- #

    def _time_shift(self, wav: torch.Tensor) -> torch.Tensor:
        shift = random.randint(-self.max_shift, self.max_shift)
        return torch.roll(wav, shifts=shift)

    def _speed(self, wav: torch.Tensor) -> torch.Tensor:
        factor = random.choice([0.9, 1.1])
        new_len = int(round(wav.shape[-1] / factor))
        wav = F.resample(wav, SR, int(SR * factor))
        return pad_or_trim(wav, new_len)

    def _gain(self, wav: torch.Tensor) -> torch.Tensor:
        gain_db = random.uniform(-6, 6)
        return wav * math.pow(10.0, gain_db / 20)

    def _rir(self, wav: torch.Tensor) -> torch.Tensor:
        rir = random.choice(self.rir_bank)
        wav = torch.nn.functional.conv1d(
            wav.unsqueeze(0), rir.flip(-1), padding=rir.numel() - 1
        ).squeeze(0)
        return wav

    # --------------------------------------------------------------------- #

    def __call__(self, wav: torch.Tensor) -> torch.Tensor:
        """Apply a random chain of augmentations."""
        wav = pad_or_trim(wav)

        # 1. always time-shift (strong prior for SC-V3)
        wav = self._time_shift(wav)

        # 2. background noise 80 % of the time
        if random.random() < 0.8 and self.bg_bank:
            noise = random.choice(self.bg_bank)
            snr = random.uniform(0, 20)
            wav = snr_mix(wav, noise, snr)

        # 3. speed perturbation (p=0.5) – implicitly applies small pitch change
        if random.random() < 0.5:
            wav = self._speed(wav)

        # 4. random gain (p=0.5)
        if random.random() < 0.5:
            wav = self._gain(wav)

        # 5. room impulse response (p=0.3)
        if self.rir_bank and random.random() < 0.3:
            wav = self._rir(wav)

        # 6. 10 % probability of dropout section (packet loss)
        if random.random() < 0.1:
            drop_len = int(0.1 * wav.numel())
            start = random.randint(0, wav.numel() - drop_len)
            wav[start : start + drop_len] = 0.0

        return pad_or_trim(wav)

# --------------------------------------------------------------------------- #
#                         spectrogram-domain aug                              #
# --------------------------------------------------------------------------- #

class SpecAug:
    """Classic SpecAugment with two frequency & two time masks."""

    def __init__(self, n_freq_masks: int = 2, n_time_masks: int = 2):
        self.f_masks = torch.nn.ModuleList([FrequencyMasking(8) for _ in range(n_freq_masks)])
        self.t_masks = torch.nn.ModuleList([TimeMasking(12) for _ in range(n_time_masks)])

    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        for m in self.f_masks:
            spec = m(spec)
        for m in self.t_masks:
            spec = m(spec)
        return spec

# --------------------------------------------------------------------------- #
#                        convenience: build pipelines                         #
# --------------------------------------------------------------------------- #

def build_default_augs(
    bg_bank: Sequence[torch.Tensor],
    rir_bank: Optional[Sequence[torch.Tensor]] = None,
) -> list[Callable]:
    """Return a list you can plug into `TFDatasetAdapter`."""
    return [
        WaveformAug(bg_bank, rir_bank),   # waveform domain
        SpecAug(),                        # spectrogram domain (after MFCC/log-Mel)
    ]

# --------------------------------------------------------------------------- #
# Example usage inside your data loader                                      #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    # quick smoke test
    wav = torch.randn(FIXED_LEN)
    bg_noise = [torch.randn(FIXED_LEN) for _ in range(4)]
    aug = WaveformAug(bg_noise)

    out = aug(wav)
    print(out.shape)  # (16000,)