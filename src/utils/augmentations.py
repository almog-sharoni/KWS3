import numpy as np
import librosa
import random

FREQUENCY = 16000
DURATION = 16000
def add_time_shift_and_align(audio, max_shift_in_ms=100):
    # randomly shift the audio by at most max_shift_in_ms
    max_shift = (max_shift_in_ms * FREQUENCY) // 1000
    time_shift = np.random.randint(0, max_shift)
    future = np.random.randint(0, 2)

    if future == 0:
        audio = np.pad(audio[time_shift:], (0, time_shift), 'constant')
    else:
        audio = np.pad(audio[:-time_shift], (time_shift, 0), 'constant')

    # Ensure the audio tensor has the correct length
    if len(audio) < DURATION:
        audio = np.pad(audio, (DURATION - len(audio), 0), 'constant')

    return audio[:DURATION]

def add_noise(audio, noise_level=0.005):
    noise = np.random.randn(len(audio)) * noise_level
    return audio + noise

def change_pitch(audio, sample_rate=FREQUENCY, pitch_factor=0.2):
    # Change the pitch of the audio
    return librosa.effects.pitch_shift(y=audio, sr=sample_rate, n_steps=pitch_factor)

def change_speed(audio, speed_factor=1.1):
    # Change the speed of the audio
    audio = librosa.effects.time_stretch(audio, rate=speed_factor)
    
    # Ensure the audio tensor has the correct length
    if len(audio) > DURATION:
        return audio[:DURATION]
    else:
        return np.pad(audio, (0, DURATION - len(audio)), 'constant')

def add_random_volume(audio, vol_range=(0.8, 1.2)):
    # Randomly change the volume of the audio
    volume_change = np.random.uniform(vol_range[0], vol_range[1])
    return audio * volume_change

def add_silence(audio, silence):
    return audio + silence


def apply_augmentations(audio,silence_ds):
    # Apply a random selection of augmentations
    if random.random() < 0.5:
        audio = add_time_shift_and_align(audio)

    audio = add_noise(audio,silence_ds)

    
    return audio