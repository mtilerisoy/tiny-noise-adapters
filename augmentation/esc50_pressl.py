import glob as gb
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import os


from src.util import get_annotations, get_individual_cycles_librosa, get_entire_signal_librosa


SR = 16000
data_dir = "datasets/ESC-50/"

# for pretraining
def preprocess_cycle_spectrogram(input_sec=2):
    sound_dir_loc = np.array(gb.glob("datasets/icbhi/ICBHI_final_database/*.wav"))
    annotation_dict = get_annotations("cycle", "datasets/icbhi/ICBHI_final_database")
    cycles_npy_names = []
    train_test = []

    cycle_path = data_dir + "cycle_spec_pad2_npy/" 
    if not os.path.exists(cycle_path): os.makedirs(cycle_path)
    
    valid_data, invalid_data = 0, 0
    for i in tqdm(range(sound_dir_loc.shape[0])):
        filename = sound_dir_loc[i].strip().split('.')[0]
        fileID = filename.split('/')[-1].split('.')[0]
        userID = filename.split('/')[-1].split('_')[0]
        
        # MTI modified these
        try:
            file_split = splits_data["group"][splits_data.fileID == fileID].values[0]
        except IndexError:
            print(f"File {filename} not found in splits_data. Skipping.")
            continue

        sample_data = get_individual_cycles_librosa('cycle', annotation_dict[fileID], "datasets/icbhi/ICBHI_final_database", fileID, SR, 2)
        
        j = 0
        for audio, label in sample_data:
            
            j += 1
            # get spectrogram if longer than 2s
            data = get_entire_signal_librosa("",  "", spectrogram=True, input_sec=input_sec, pad=False, from_cycle=True, yt=audio)
            if data is None:
                invalid_data += 1
                continue
            cycles_npy_names.append(cycle_path + fileID + "cycle" + str(j))
            np.save(cycle_path + fileID + "cycle" + str(j), data)
            valid_data += 1
            train_test.append(file_split)

    print(len(cycles_npy_names), len(train_test))
    np.save(data_dir + "cycle_spec_pad2_name.npy", cycles_npy_names)
    print("valid_data", valid_data, "invalid_data", invalid_data) # valid_data 5024 invalid_data 1874
    np.save("datasets/icbhi/cycle_spec_split.npy", train_test)


def preprocess_entire_spectrogram(input_sec=8):
    sound_dir_loc = np.array(gb.glob("datasets/ESC-50/audio/*.wav"))
    train_test = []
    filename_list = []
    invalid_data = 0
    classes = []
    
    for i in tqdm(range(sound_dir_loc.shape[0])):
        filename = sound_dir_loc[i].strip().split('.')[0]
        fileID = filename.split('/')[-1]
        classes.append(filename.split('/')[-1].split('-')[3])
        
        # userID = filename.split('/')[-1].split('_')[0]
        # label = labels_data["class"][labels_data.userID == userID].values[0]
        
        # MTI modified these
        data = get_entire_signal_librosa("", filename.split('.')[0], spectrogram=True, input_sec=input_sec, pad=True)

        if data is None:
            invalid_data += 1
            continue

        np.save("datasets/ESC-50/entire_spec_npy_8000/" + fileID + ".npy", data)
        filename_list.append("datasets/ESC-50/entire_spec_npy_8000/" + fileID)
        np.save("datasets/ESC-50/entire_spec_filenames_8000.npy", filename_list)
    print("invalid_data", invalid_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_sec", type=int, default=8)
    args = parser.parse_args()

    preprocess_cycle_spectrogram()
    preprocess_entire_spectrogram(input_sec=args.input_sec)


