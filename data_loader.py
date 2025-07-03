# data_loader.py

import os
import yaml
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import random

from noise_injector import NoiseInjector
from utils import load_audio, process_audio_length, pre_process_audio_mel_t

TARGET_NOISE_FILES = [
    "4-119647-A-48",
    "4-163697-A-13",
    "2-80844-A-13",
    "1-18527-A-44",
    "1-54918-A-14"
    # "running_tap.wav" # Add more noise files here
]

class NoisyDataGenerator(Dataset):
    """
    A PyTorch Dataset for on-the-fly generation of noisy audio samples.
    """
    def __init__(self, config_path: str, split: str):
        super().__init__()
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.injector = NoiseInjector(config_path)

        self.sr = self.config['sample_rate']
        self.duration = self.config['target_duration_sec']
        
        df_splits = pd.read_csv(self.config['clean_data_split_path'])
        # store the filenames which might not have extensions
        self.file_list = df_splits[df_splits['split'] == split]['audio_filename'].tolist()
        self.clean_data_path = self.config['clean_data_path']

        self.noise_data_path = self.config['noise_data_path']
        self.ambient_noise_files = [f for f in os.listdir(self.noise_data_path) if f.endswith('.wav')]
        self.snr_levels = self.config['snr_levels_db']
        
        self.artefact_specs = [
            {'type': 'low_pass', 'cutoff_hz': 2000, 'order': 5},
            {'type': 'low_pass', 'cutoff_hz': 4000, 'order': 5},
            {'type': 'clipping', 'gain_db': 6.0},
            {'type': 'clipping', 'gain_db': 12.0},
        ]
        
        gen_settings = self.config.get('generator_settings', {})
        self.p_clean = gen_settings.get('clean_sample_prob', 0.1)
        self.p_artefact = gen_settings.get('artefact_prob', 0.45)
        self.p_noise = gen_settings.get('ambient_noise_prob', 0.45)
        
        print(f"Initialized NoisyDataGenerator for '{split}' split with {len(self.file_list)} files.")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # load clean recording
        csv_filename = self.file_list[idx]
        base_name = os.path.splitext(csv_filename)[0]
        load_filename = f"{base_name}.wav"

        clean_path = os.path.join(self.clean_data_path, load_filename)
        clean_audio_raw = load_audio(clean_path, sr=self.sr)
        
        if clean_audio_raw is None:
            # if a file is missing, return a silent sample
            print(f"Warning: Could not load {clean_path}, returning silent audio for this item.")
            silent_len = int(self.duration * self.sr)
            silent_audio = np.zeros((1, silent_len), dtype=np.float32)
            return torch.from_numpy(silent_audio), torch.from_numpy(silent_audio)
        
        clean_audio_processed = process_audio_length(clean_audio_raw, self.duration, self.sr)
        
        # randomly choose an augmentation type
        augmentation_options = ['clean', 'artefact', 'noise']
        augmentation_probs = [self.p_clean, self.p_artefact, self.p_noise]
        chosen_augment = random.choices(population=augmentation_options, weights=augmentation_probs, k=1)
        print(f"Chosen augmentation for {load_filename}: {chosen_augment[0]}")
        
        # CLEAN AUDIO
        if chosen_augment == ['clean']:
            print(f"Returning clean audio for {load_filename}.")

            clean_audio_processed = pre_process_audio_mel_t(clean_audio_processed, f_max=8000)
            return torch.from_numpy(clean_audio_processed), torch.from_numpy(clean_audio_processed)
        
        # ARTEFACT
        elif chosen_augment == ['artefact']:
            print(f"Applying artefact '{spec['type']}' to {load_filename}.")
            
            spec = random.choice(self.artefact_specs)
            noisy_audio = self.injector.add_artefact(clean_audio_raw, spec)
            if spec['type'] != 'clipping':
                noisy_audio = pre_process_audio_mel_t(noisy_audio, f_max=8000)
                clean_audio_processed = pre_process_audio_mel_t(clean_audio_processed, f_max=8000)
            return torch.from_numpy(noisy_audio), torch.from_numpy(clean_audio_processed)
        
        # AMBIENT NOISE
        else:
            # randomly select a noise file
            if TARGET_NOISE_FILES:
                noise_filename = TARGET_NOISE_FILES[0]
                TARGET_NOISE_FILES.remove(noise_filename)  # remove to avoid repetition
            else:
                noise_filename = random.choice(self.ambient_noise_files)

            print(f"Injecting noise from {noise_filename} into {load_filename}.")
            noise_path = os.path.join(self.noise_data_path, noise_filename)
            noise_audio_raw = load_audio(noise_path, sr=self.sr)
            
            # if a noise file fails, default to a clean sample
            if noise_audio_raw is None:    
                print(f"Warning: Could not load {noise_path}, using clean audio as noisy sample.")
                clean_audio_processed = pre_process_audio_mel_t(clean_audio_processed, f_max=8000)
                return torch.from_numpy(clean_audio_processed), torch.from_numpy(clean_audio_processed)
            else:
                snr = random.choice(self.snr_levels)
                noisy_audio = self.injector.add_noise(clean_audio_raw, noise_audio_raw, snr)
                noisy_audio = pre_process_audio_mel_t(noisy_audio, f_max=8000)
                clean_audio_processed = pre_process_audio_mel_t(clean_audio_processed, f_max=8000)

                noise_audio_raw = process_audio_length(noise_audio_raw, self.duration, self.sr)
                noise_audio_raw = pre_process_audio_mel_t(noise_audio_raw, f_max=8000)

                return torch.from_numpy(noisy_audio), torch.from_numpy(clean_audio_processed), torch.from_numpy(noise_audio_raw)