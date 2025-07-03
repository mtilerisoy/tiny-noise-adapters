# generate_test_set.py

import os
import yaml
import pandas as pd
from tqdm import tqdm
import shutil
import random

from noise_injector import NoiseInjector
from utils import load_audio, save_audio, peak_normalize

def generate_test_set(config_path: str):
    """
    Generates a fixed, static test set based on the configuration file.

    This script iterates through all clean test files and applies every
    configured ambient noise and synthetic artefact.
    """

    # get the configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    injector = NoiseInjector(config_path)

    # get the injection probabilities
    gen_settings = config.get('generator_settings', {})
    clean_prob = gen_settings.get('clean_sample_prob', 0.0)
    artefact_prob = gen_settings.get('artefact_prob', 0.0)
    ambient_prob = gen_settings.get('ambient_noise_prob', 0.0)
    
    # define the paths
    clean_data_path = config['clean_data_path']
    split_file_path = config['clean_data_split_path']
    noise_data_path = config['noise_data_path']
    output_path = config['output_path']

    # dataset identifier
    output_identifier = config.get('output_identifier', '')
    
    # create output directories
    wav_output_dir = os.path.join(output_path, 'noisy_wav')
    os.makedirs(wav_output_dir, exist_ok=True)

    # load the split file
    df_splits = pd.read_csv(split_file_path)

    # filter only test samples for now
    df_test = df_splits[df_splits['split'] == 'test']
    print(f"Found {len(df_test)} files in the test set.")
    
    # list noise recordings
    ambient_noise_files = [f for f in os.listdir(noise_data_path) if f.endswith(('.wav', '.flac', '.mp3'))]
    print(f"Found {len(ambient_noise_files)} ambient noise files.")
    
    # define artefact specs
    artefact_specs = [
        {'type': 'low_pass', 'cutoff_hz': 2000, 'order': 5},
        {'type': 'low_pass', 'cutoff_hz': 4000, 'order': 5},
        {'type': 'clipping', 'gain_db': 6.0},
        {'type': 'clipping', 'gain_db': 12.0},
    ]

    # inject noise and artefacts
    for _, row in tqdm(df_test.iterrows(), total=df_test.shape[0], desc="Processing test files"):
        csv_filename = row['audio_filename']
        original_name = os.path.splitext(csv_filename)[0]
        clean_filename = original_name + ".wav"
        clean_file_path = os.path.join(clean_data_path, clean_filename)
        
        clean_audio = load_audio(clean_file_path, sr=config['sample_rate'])
        if clean_audio is None:
            print(f"Warning: Skipping file, could not load {clean_file_path}")
            continue

        # keep the sample clean
        if random.random() < clean_prob:
            clean_gt_dir = os.path.join(wav_output_dir, "clean_ground_truth")
            os.makedirs(clean_gt_dir, exist_ok=True)
            shutil.copy2(clean_file_path, os.path.join(clean_gt_dir, original_name + f"__{output_identifier}.wav"))

        # inject ambient noise
        if random.random() < ambient_prob:
            noise_filename = random.choice(ambient_noise_files)
            noise_name = os.path.splitext(noise_filename)[0]
            noise_file_path = os.path.join(noise_data_path, noise_filename)
            noise_audio = load_audio(noise_file_path, sr=config['sample_rate'])
            if noise_audio is not None:
                snr = random.choice(config['snr_levels_db'])
                noisy_audio = injector.add_noise(clean_audio, noise_audio, snr)
                normalized_audio = peak_normalize(noisy_audio)
                output_filename = f"{original_name}_ambient_{noise_name}_{snr}dB__{output_identifier}.wav"
                output_dir = os.path.join(wav_output_dir, "ambient_noise", f"snr_{snr}dB")
                os.makedirs(output_dir, exist_ok=True)
                save_audio(os.path.join(output_dir, output_filename), normalized_audio, config['sample_rate'])


        # inject synthetic artefacts
        if random.random() < artefact_prob:
            spec = random.choice(artefact_specs)
            artefact_audio = injector.add_artefact(clean_audio, spec)
            if spec['type'] != 'clipping':
                artefact_audio = peak_normalize(artefact_audio)
            if spec['type'] == 'low_pass':
                desc = f"cutoff{spec['cutoff_hz']}Hz"
            elif spec['type'] == 'clipping':
                desc = f"gain{spec['gain_db']}dB"
            else:
                desc = "artefact"
            output_filename = f"{original_name}_{spec['type']}_{desc}_NA__{output_identifier}.wav"
            output_dir = os.path.join(wav_output_dir, "synthetic_artefacts", spec['type'])
            os.makedirs(output_dir, exist_ok=True)
            save_audio(os.path.join(output_dir, output_filename), artefact_audio, config['sample_rate'])

    print("\nTest set generation complete.")

if __name__ == '__main__':
    generate_test_set('config.yaml')