def add_time_shift_and_align(audio, shift_range=0.2):
    # Function to add time shifts to audio data
    shift = np.random.uniform(-shift_range, shift_range) * len(audio)
    shifted_audio = np.roll(audio, int(shift))
    return shifted_audio

def add_silence(audio, silence_duration=0.5, sample_rate=16000):
    # Function to add silence to audio data
    silence = np.zeros(int(silence_duration * sample_rate))
    return np.concatenate((audio, silence))