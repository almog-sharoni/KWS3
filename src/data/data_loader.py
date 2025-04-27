from __future__ import annotations
import tensorflow as tf
import tensorflow_datasets as tfds
from torch.utils.data import DataLoader, Dataset
from librosa.feature import mfcc, delta
import numpy as np
import torch
import src.utils.augmentations
import random
import os
import librosa

import os
import requests
import librosa
import numpy as np

import tarfile


import random
from typing import Sequence, Optional, Callable

import torch
from torch.utils.data import Dataset
import torchaudio
import torchaudio.functional as F

from src.utils.augmentations import pad_or_trim  # util defined earlier


def download_and_extract_urbansound8k(dataset_url, extract_path):
    # Define the tar.gz file path and the expected dataset directory
    tar_path = os.path.join(extract_path, 'UrbanSound8K.tar.gz')
    dataset_dir = '/shared_data/KWS3/UrbanSound8K'
    
    # Check if the dataset directory already exists
    if os.path.exists(dataset_dir):
        print("UrbanSound8K dataset already exists. Skipping download and extraction.")
        return
    
    # Download the dataset if it doesn't exist
    if not os.path.exists(tar_path):
        print("Downloading UrbanSound8K dataset...")
        response = requests.get(dataset_url, stream=True)
        with open(tar_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")
    
    # Extract the dataset
    print("Extracting UrbanSound8K dataset...")
    with tarfile.open(tar_path, 'r:gz') as tar_ref:
        tar_ref.extractall(extract_path)
    print("Extraction complete.")

def load_bg_noise_dataset():
    # URL to download the UrbanSound8K dataset
    dataset_url = "https://zenodo.org/records/1203745/files/UrbanSound8K.tar.gz"
    dataset_path = '/shared_data/KWS3/UrbanSound8K'
    
    # Ensure the dataset is downloaded and extracted
    download_and_extract_urbansound8k(dataset_url, '.')

    bgNoise = []

    # Loop through all folds and load audio files
    for fold in range(1, 10):  # Folds are from fold1 to fold9
        fold_path = os.path.join(dataset_path, 'audio', f'fold{fold}')
        
        # List all audio files in the current fold
        for file_name in os.listdir(fold_path):
            if file_name.endswith('.wav'):  # Process only .wav files
                file_path = os.path.join(fold_path, file_name)
                
                # Load the audio file
                audio, sample_rate = librosa.load(file_path, sr=None)
                # Normalize audio
                audio = audio / np.max(np.abs(audio))
                bgNoise.append(audio)
    return bgNoise

def load_speech_commands_dataset(version=3,reduced=False):
    """Load the Speech Commands dataset using TensorFlow Datasets."""
    with tf.device('/cpu:0'):
        ds, info = tfds.load(f'speech_commands:0.0.{version}', with_info=True, as_supervised=True, shuffle_files=False)
        train_ds = ds['train']
        val_ds = ds['validation']
        test_ds = ds['test']

        train_silence = train_ds.filter(lambda x, y: y == 10)

    # Filter out the 'unknown' and 'silence' labels
    # If label = 10,11 drop
    if reduced:
        train_ds = train_ds.filter(lambda x, y: y != 10 and y != 11)
        val_ds = val_ds.filter(lambda x, y: y != 10 and y != 11)
        test_ds = test_ds.filter(lambda x, y: y != 10 and y != 11)
    return train_ds, val_ds, test_ds, train_silence, info

# Define the dataset adapter:
# Define the dataset adapter:
# tf_dataset_adapter.py

# --------------------------------------------------------------------------- #
#                                main adapter                                 #
# --------------------------------------------------------------------------- #

class TFDatasetAdapter(Dataset):
    """
    Wrap a TensorFlow-Dataset-style Speech Commands split for PyTorch training.

    Parameters
    ----------
    tf_dataset :  iterable yielding (audio_tf, label_tf)
    waveform_augs : list[Callable[[torch.Tensor], torch.Tensor]]
        Augmentations that expect a 1-D waveform (length == fixed_length).
    spec_augs : list[Callable[[torch.Tensor], torch.Tensor]]
        Augmentations that expect a 2-D spectrogram / MFCC.
    """

    def __init__(
        self,
        tf_dataset,
        *,
        fixed_length: int = 16_000,
        sample_rate: int = 16_000,
        n_mfcc: int = 13,
        n_fft: int = 400,
        hop_length: int = 160,
        n_mels: int = 40,
        waveform_augs: Optional[Sequence[Callable]] = None,
        spec_augs: Optional[Sequence[Callable]] = None,
        derivative: bool = True,
        compute_mfcc: bool = True,
    ):
        self.samples = list(tf_dataset)          # cache to Python list
        self.fixed_length = fixed_length
        self.sample_rate = sample_rate
        self.waveform_augs = list(waveform_augs or [])
        self.spec_augs = list(spec_augs or [])
        self.compute_mfcc = compute_mfcc
        self.derivative = derivative

        # build MFCC transform once
        self.mfcc_xform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs=dict(
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
                f_min=0,
                f_max=sample_rate // 2,
            ),
        )

    # ------------------------------ dataset API ---------------------------- #

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        audio_tf, label_tf = self.samples[idx]

        # TensorFlow -> torch
        wav = torch.from_numpy(audio_tf.numpy().astype("float32"))
        wav = wav / (wav.abs().max().clamp(min=1e-9))          # normalise
        if wav.ndim > 1:                                       # squeeze channels
            wav = wav.squeeze()

        wav = pad_or_trim(wav, self.fixed_length)              # 1-s exactly

        # ---------------- waveform-level augmentations ------------------- #
        for aug in self.waveform_augs:
            wav = aug(wav)

        # ---------------- MFCC (or raw waveform) ------------------------ #
        if self.compute_mfcc:
            spec = self.mfcc_xform(wav.unsqueeze(0))           # (1, n_mfcc, T)
            spec = spec.squeeze(0)                             # (n_mfcc, T)

        if self.derivative:
            delta1 = F.compute_deltas(spec)            # Δ
            delta2 = F.compute_deltas(delta1)          # Δ²
            spec   = torch.cat([spec, delta1, delta2], dim=0)
        else:
            spec = wav.unsqueeze(0)   # keep extra dimension to look 2-D to spec_augs

        # ---------------- spectrogram-level augmentations --------------- #
        for aug in self.spec_augs:
            spec = aug(spec)

        return spec.float(), torch.tensor(label_tf.numpy(), dtype=torch.long)


# class TFDatasetAdapter(Dataset):
#     def __init__(self, tf_dataset, silence_dataset, fixed_length, n_mfcc, n_fft, hop_length, n_mels, augmentation=False, derivative=True, noise_level=0.3):
#         self.tf_dataset = tf_dataset
#         self.data = list(tf_dataset)
#         self.silence_data = list(silence_dataset) if silence_dataset is not None else None
#         self.fixed_length = fixed_length
#         self.n_mfcc = n_mfcc
#         self.n_fft = n_fft
#         self.hop_length = hop_length
#         self.n_mels = n_mels
#         self.augmentation = augmentation
#         self.derivative = derivative
#         self.noise_level = noise_level

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         audio, label = self.data[idx]
#         audio = audio.numpy()

#         # Convert to float
#         audio = audio.astype(np.float32)

#         # Ensure the audio tensor has the correct shape (1D array)
#         if audio.ndim > 1:
#             audio = np.squeeze(audio)
        
#         # Add noise from silence data
#         if self.silence_data:
#             silence_audio, _ = random.choice(self.silence_data)
#             silence_audio = silence_audio.numpy().astype(np.float32)

#             # Trim or pad silence to match the audio length
#             if len(silence_audio) < len(audio):
#                 silence_audio = np.pad(silence_audio, (0, len(audio) - len(silence_audio)), mode='constant')
#             else:
#                 silence_audio = silence_audio[:len(audio)]

#             # Add silence as noise to the original audio
#             audio = audio + self.noise_level * silence_audio

#         # Apply augmentations if any
#         if self.augmentation:
#             for aug in self.augmentation:
#                 audio = aug(audio)

#         # Pad or trim the audio to the fixed length
#         if len(audio) < self.fixed_length:
#             audio = np.pad(audio, (0, self.fixed_length - len(audio)), mode='constant')
#         else:
#             audio = audio[:self.fixed_length]

#         # Create MFCCs from an audio tensor using Librosa.
#         audio = audio.astype(np.float32)
#         MFCC = mfcc(y=audio, sr=16000, n_mfcc=self.n_mfcc, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels)
#         if self.derivative:
#             # Create MFCC second, first order delta
#             MFCC_delta = delta(MFCC)
#             MFCC_delta2 = delta(MFCC, order=2)

#             # Stack the three MFCCs together
#             MFCC = np.vstack([MFCC, MFCC_delta, MFCC_delta2])

#         # Remove extra dimension if it exists
#         if MFCC.ndim == 3:
#             MFCC = MFCC.squeeze(-1)

#         return torch.tensor(MFCC, dtype=torch.float32), torch.tensor(label.numpy(), dtype=torch.long)    



