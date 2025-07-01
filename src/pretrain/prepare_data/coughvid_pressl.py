import argparse
import numpy as np
from tqdm import tqdm
from src.util import get_entire_signal_librosa
import os

# for pretraining


def preprocess_spectrogram_SSL(modality="breath", input_sec=8):

    train_uids = np.load(
        "datasets/coughvid/coughvid__train_uuids.npy", allow_pickle=True).tolist()
    val_uids = np.load(
        "datasets/coughvid/coughvid__val_uuids.npy", allow_pickle=True).tolist()
    uids = train_uids + val_uids
    print('training:', len(uids))

    invalid_data = 0

    filename_list = []

    file_dir = os.listdir('datasets/coughvid/wav')
    # use metadata as outer loop to enable quality check
    for file in tqdm(file_dir):

        userID = file.split('.')[0]

        if userID in uids:
            data = get_entire_signal_librosa(
                'datasets/coughvid/wav', userID, spectrogram=True, input_sec=input_sec)

            if data is None:
                invalid_data += 1
                continue

            # saving to individual npy files
            np.save("datasets/coughvid/entire_spec_npy/" +
                    userID + ".npy", data)
            filename_list.append("datasets/coughvid/entire_spec_npy/" + userID)

    np.save("datasets/coughvid/entire_spec_filenames.npy", filename_list)
    print("finished preprocessing cough: valid data", len(
        filename_list), "; invalid data", invalid_data)

# finished preprocessing cough: valid data 7179 ; invalid data 327


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_sec", type=int, default=2)
    args = parser.parse_args()

    preprocess_spectrogram_SSL(input_sec=args.input_sec)
