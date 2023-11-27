import os
import pprint

import numpy as np
import pandas as pd
from PIL import Image

df = pd.read_pickle("study2DataFrame.pkl")

all_subjects = df["Subject"].unique()

spectogram_map = {
    256: [8, 64, 128, 250],
    512: [0, 256, 511],
}

TEST_RAW_DIR = "test"
RETEST_RAW_DIR = "retest"
NPY_DATA_DIR = "npy_datasets"


def gen_npy_data(raw_dir):
    for window_size, overlaps in spectogram_map.items():
        for overlap in overlaps:
            npy_dataset = []
            png_dataset = []
            for i, subject in enumerate(all_subjects):
                files = os.listdir(raw_dir)
                for test_file in files:
                    # search for npy files that match this pattern "{subject}_spectogram_{0 or 1}_{window_size}_{overlap}"
                    if (
                        subject + "_" in test_file
                        and f"{window_size}_{overlap}" in test_file
                        and test_file.endswith(".npy")
                    ):
                        # load the file
                        npy_file = np.load(f"{raw_dir}/{test_file}")
                        npy_dataset.append((npy_file, i))

                    # load the png file
                    if (
                        subject + "_" in test_file
                        and f"{window_size}_{overlap}" in test_file
                        and test_file.endswith(".png")
                    ):
                        png_img = Image.open(f"{raw_dir}/{test_file}").convert('L')
                        png_img.thumbnail((32, 32),Image.ANTIALIAS)
                        png_file = np.array(png_img)
                        # print('png_file.shape', png_file.shape)
                        # print(png_file.min(), png_file.max())
                        png_dataset.append((png_file, i))
            npy_dataset = np.array(npy_dataset)
            # save the npy dataset
            np.save(
                f"{NPY_DATA_DIR}/{raw_dir}_{window_size}_{overlap}.npy", npy_dataset
            )
            # save the png dataset
            np.save(
                f"{NPY_DATA_DIR}/{raw_dir}_{window_size}_{overlap}_png.npy",
                np.array(png_dataset),
            )


gen_npy_data(TEST_RAW_DIR)
gen_npy_data(RETEST_RAW_DIR)
