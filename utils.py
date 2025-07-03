# utils.py

import librosa
import numpy as np
import soundfile as sf
import os
from typing import Optional

def load_audio(file_path: str, sr: int, mono: bool = True) -> Optional[np.ndarray]:
    """
    Loads an audio file and converts it to the specified sample rate.

    Args:
        file_path (str): Path to the audio file.
        sr (int): The target sample rate to resample to.
        mono (bool): If True, converts the audio to mono. Defaults to True.

    Returns:
        Optional[np.ndarray]: A NumPy array of the audio data as float32.
                              Returns None if the file cannot be loaded.
    """
    try:
        file_path = file_path if file_path.endswith('.wav') else file_path + '.wav'
        audio_data, _ = librosa.load(file_path, sr=sr, mono=mono)
        return audio_data
    except Exception as e:
        print(f"Error loading audio file {file_path}: {e}")
        return None

def save_audio(file_path: str, audio_data: np.ndarray, sr: int):
    """
    Saves a NumPy audio array to a WAV file.

    This function creates the output directory if it does not exist.

    Args:
        file_path (str): The path where the audio file will be saved.
        audio_data (np.ndarray): The NumPy array containing the audio data.
        sr (int): The sample rate of the audio data.
    """
    try:
        output_dir = os.path.dirname(file_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        sf.write(file_path, audio_data, sr)
    except Exception as e:
        print(f"Error saving audio file to {file_path}: {e}")

def process_audio_length(audio_data: np.ndarray, target_duration_sec: float, sample_rate: int) -> np.ndarray:
    """
    Standardizes the length of an audio signal by truncating or padding.

    Args:
        audio_data (np.ndarray): The input audio signal.
        target_duration_sec (float): The desired duration of the audio in seconds.
        sample_rate (int): The sample rate of the audio.

    Returns:
        np.ndarray: The processed audio signal with the exact target length.
    """
    target_length_samples = int(target_duration_sec * sample_rate)
    current_length_samples = audio_data.shape[0]

    # Truncate the audio if it is longer than the target length
    if current_length_samples > target_length_samples:
        processed_audio = audio_data[:target_length_samples]

    # Pad the audio with zeros (silence)
    elif current_length_samples < target_length_samples:
        padding_needed = target_length_samples - current_length_samples
        padding_array = np.zeros(padding_needed, dtype=audio_data.dtype)
        processed_audio = np.concatenate([audio_data, padding_array])

    # The audio is already the correct length
    else:
        processed_audio = audio_data

    return processed_audio

def peak_normalize(audio: np.ndarray) -> np.ndarray:
    """
    Normalizes audio to a peak value of 1.0, preventing clipping.

    Args:
        audio (np.ndarray): The audio signal to normalize.

    Returns:
        np.ndarray: The normalized audio signal.
    """
    max_peak = np.max(np.abs(audio))
    
    # Avoid division by zero for silent audio
    if max_peak > 0:
        return audio / max_peak
    
    return audio

def pre_process_audio_mel_t(audio, sample_rate=16000, n_mels=64, f_min=50, f_max=2000, nfft=1024, hop=512):
    S = librosa.feature.melspectrogram(
        y=audio, sr=sample_rate, n_mels=n_mels, fmin=f_min, fmax=f_max, n_fft=nfft, hop_length=hop)
    # convert scale to dB from magnitude
    S = librosa.power_to_db(S, ref=np.max)
    if S.max() != S.min():
        mel_db = (S - S.min()) / (S.max() - S.min())
    else:
        mel_db = S
        print("warning in producing spectrogram!")

    return mel_db