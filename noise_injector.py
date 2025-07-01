# noise_injector.py

import numpy as np
import yaml
from scipy.signal import butter, filtfilt
from typing import Dict, Any

from utils import process_audio_length

class NoiseInjector:
    """
    A class to apply various types of noise and artefacts to audio signals.
    """

    def __init__(self, config_path: str):
        """Initializes the NoiseInjector with parameters from the config file.
        
        Args:
            config_path (str): Path to the configuration YAML file.
        """

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.sr = self.config['sample_rate']
        self.duration = self.config['target_duration_sec']
        self.seed = self.config['random_seed']
        self.rng = np.random.default_rng(self.seed)

    def _calculate_rms(self, audio: np.ndarray) -> float:
        """Calculates the Root Mean Square of an audio signal.
        Args:
            audio (np.ndarray): The audio signal to process.
        Returns:
            float: The RMS value of the audio signal.
        """

        return np.sqrt(np.mean(audio**2))

    def add_noise(self, clean_audio: np.ndarray, noise_audio: np.ndarray, snr_db: float) -> np.ndarray:
        """
        Adds ambient noise to a clean audio signal at a specified SNR.
        This function returns an wpverlayed signal without being normalized.

        Args:
            clean_audio (np.ndarray): The clean audio signal.
            noise_audio (np.ndarray): The noise audio signal.
            snr_db (float): The desired Signal-to-Noise Ratio in dB.
        Returns:
            np.ndarray: The noisy audio signal with the specified SNR.
        """

        clean_proc = process_audio_length(clean_audio, self.duration, self.sr)
        noise_proc = process_audio_length(noise_audio, self.duration, self.sr)

        rms_clean = self._calculate_rms(clean_proc)
        rms_noise = self._calculate_rms(noise_proc)
        
        if rms_clean == 0 or rms_noise == 0:
            return clean_proc

        snr_linear = 10**(snr_db / 20.0)
        required_noise_rms = rms_clean / snr_linear
        scaling_factor = required_noise_rms / rms_noise

        noisy_audio = clean_proc + (noise_proc * scaling_factor)
        return noisy_audio

    def add_artefact(self, clean_audio: np.ndarray, artefact_spec: Dict[str, Any]) -> np.ndarray:
        """Applies a synthetic sonic artefact to a clean audio signal.
        
        Args:
            clean_audio (np.ndarray): The clean audio signal.
            artefact_spec (Dict[str, Any]): A dictionary specifying the type and parameters of the artefact.
        
        Returns:
            np.ndarray: The audio signal with the applied artefact."""

        clean_proc = process_audio_length(clean_audio, self.duration, self.sr)
        artefact_type = artefact_spec.get('type')
        
        if artefact_type == 'low_pass':
            cutoff = artefact_spec.get('cutoff_hz', 2000)
            order = artefact_spec.get('order', 5)
            nyquist = 0.5 * self.sr
            normal_cutoff = cutoff / nyquist
            b, a = butter(order, normal_cutoff, btype='low', analog=False)
            artefact_audio = filtfilt(b, a, clean_proc)
            return artefact_audio

        elif artefact_type == 'clipping':
            gain_db = artefact_spec.get('gain_db', 6.0)
            gain_linear = 10**(gain_db / 20.0)
            gained_audio = clean_proc * gain_linear
            clipped_audio = np.clip(gained_audio, -1.0, 1.0)
            return clipped_audio
        
        else:
            raise ValueError(f"Unsupported artefact type: {artefact_type}")