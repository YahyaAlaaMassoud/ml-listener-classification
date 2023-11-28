import os
import pprint

import numpy as np
import pandas as pd
from PIL import Image
from data_utils import normalize_min_max_v2, standardize

df = pd.read_pickle("study2DataFrame.pkl")

all_subjects = df["Subject"].unique()

spectogram_map = {
    256: [8, 64, 128, 250],
    512: [0, 256, 511],
}


def gen_multires_npy_data(raw_dir, window_size, overlaps):
    # you should generate a numpy dataset where each row is a tuple
    # the tuple is (X, y), y is the subject number
    # X is the spectogram png for each of the overlaps in the overlaps list
    # X should be a numpy array of shape ((32, 32), (32, 32)...) where the number of
    # (32, 32) is the number of overlaps in the overlaps list
    
    multires_datast = []
    for i, subject in enumerate(all_subjects):
        files = os.listdir(raw_dir)
        all_data_0 = []
        all_data_1 = []
        for test_file in files:
            if subject + "_" in test_file:
                for overlap in overlaps:
                    if (
                        'spectogram_0' in test_file
                        and f"_{window_size}_{overlap}." in test_file
                        and test_file.endswith(".png")
                    ):
                        spectogram_file = Image.open(f"{raw_dir}/{test_file}").convert('L')
                        spectogram_file = np.array(spectogram_file)
                        all_data_0.append(spectogram_file)
                        
                    if (
                        'spectogram_1' in test_file
                        and f"_{window_size}_{overlap}." in test_file
                        and test_file.endswith(".png")
                    ):
                        spectogram_file = Image.open(f"{raw_dir}/{test_file}").convert('L')
                        spectogram_file = np.array(spectogram_file)
                        all_data_1.append(spectogram_file)
                        
        # all_data_0.append(i)
        all_data_0 = (np.array(all_data_0), i)
        
        # all_data_1.append(i)
        all_data_1 = (np.array(all_data_1), i)
        
        multires_datast.append(all_data_0)
        multires_datast.append(all_data_1)
    
    multires_datast = np.array(multires_datast)
    print(multires_datast.shape)
    
    np.save(
        f"{NPY_DATA_DIR}/{raw_dir}_{window_size}_multires.npy", 
        multires_datast
    )
    

def gen_npy_data(raw_dir):
    for window_size, overlaps in spectogram_map.items():
        for overlap in overlaps:
            npy_dataset = []
            png_dataset = []
            multitask_dataset = []

            for i, subject in enumerate(all_subjects):
                files = os.listdir(raw_dir)
                
                all_subject_files = {}
                for test_file in files:
                    if subject + "_" in test_file:
                        if 'ampspectra_0' in test_file:
                            ampspectra_file = np.load(f"{raw_dir}/{test_file}")
                            all_subject_files['ampspectra_0'] = ampspectra_file
                        
                        if 'ampspectra_1' in test_file:
                            ampspectra_file = np.load(f"{raw_dir}/{test_file}")
                            all_subject_files['ampspectra_1'] = ampspectra_file
                            
                        if (
                            'spectogram_0' in test_file
                            and f"_{window_size}_{overlap}." in test_file
                            and test_file.endswith(".npy")
                        ):
                            spectogram_file = np.load(f"{raw_dir}/{test_file}")
                            all_subject_files['spectogram_0_npy'] = spectogram_file
                            
                        if (
                            'spectogram_1' in test_file
                            and f"_{window_size}_{overlap}." in test_file
                            and test_file.endswith(".npy")
                        ):
                            spectogram_file = np.load(f"{raw_dir}/{test_file}")
                            all_subject_files['spectogram_1_npy'] = spectogram_file
                            
                        if (
                            'spectogram_0' in test_file
                            and f"_{window_size}_{overlap}." in test_file
                            and test_file.endswith(".png")
                        ):
                            spectogram_file = Image.open(f"{raw_dir}/{test_file}").convert('L')
                            # spectogram_file.thumbnail((32, 32),Image.ANTIALIAS)
                            spectogram_file = np.array(spectogram_file)
                            all_subject_files['spectogram_0_png'] = spectogram_file
                            
                        if (
                            'spectogram_1' in test_file
                            and f"_{window_size}_{overlap}." in test_file
                            and test_file.endswith(".png")
                        ):
                            spectogram_file = Image.open(f"{raw_dir}/{test_file}").convert('L')
                            # print(f"{raw_dir}/{test_file}")
                            spectogram_file = np.array(spectogram_file)
                            all_subject_files['spectogram_1_png'] = spectogram_file
                            
                for k, v in all_subject_files.items():
                    if 'spectogram' in k and 'npy' in k:
                        npy_dataset.append((v, i))
                    if 'spectogram' in k and 'png' in k:
                        png_dataset.append((v, i))
                        
                for j in range(2):
                    multitask_dataset.append(
                        (
                            all_subject_files[f'spectogram_{j}_png'],
                            (
                                normalize_min_max_v2(all_subject_files[f'ampspectra_{j}'], 0, 1),
                                # np.log(np.array([np.max(all_subject_files[f'ampspectra_{j}']), np.argmax(all_subject_files[f'ampspectra_{j}'])])),
                                i
                            )
                        )
                    )
                        
            npy_dataset = np.array(npy_dataset)
            png_dataset = np.array(png_dataset)
            multitask_dataset = np.array(multitask_dataset)
            # print(f"multitask_dataset.shape: {multitask_dataset.shape}")
            # print(multitask_dataset[0][0].shape)
            # print(multitask_dataset[0][1][0].shape)
            # save the npy dataset
            np.save(
                f"{NPY_DATA_DIR}/{raw_dir}_{window_size}_{overlap}.npy", 
                npy_dataset
            )
            # save the png dataset
            np.save(
                f"{NPY_DATA_DIR}/{raw_dir}_{window_size}_{overlap}_png.npy",
                png_dataset,
            )
            # save the multitask dataset
            np.save(
                f"{NPY_DATA_DIR}/{raw_dir}_{window_size}_{overlap}_multitask.npy",
                multitask_dataset,
            )

TEST_RAW_DIR = "test"
RETEST_RAW_DIR = "retest"
NPY_DATA_DIR = "npy_datasets"

gen_npy_data(TEST_RAW_DIR)
gen_npy_data(RETEST_RAW_DIR)


gen_multires_npy_data(TEST_RAW_DIR, 256, [8, 64, 128, 250])
gen_multires_npy_data(TEST_RAW_DIR, 512, [0, 256, 511])

gen_multires_npy_data(RETEST_RAW_DIR, 256, [8, 64, 128, 250])
gen_multires_npy_data(RETEST_RAW_DIR, 512, [0, 256, 511])