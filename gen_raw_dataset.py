import itertools
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image
from scipy import signal
from scipy.fft import fft
from scipy.ndimage import zoom


def rescale_spectrogram(spectrogram, new_size=(64, 64)):
    # Calculate the zoom factor for each dimension
    zoom_factor = [n / o for n, o in zip(new_size, spectrogram.shape)]
    # Use scipy's zoom function to resize the spectrogram
    spectrogram_rescaled = zoom(spectrogram, zoom_factor)
    return spectrogram_rescaled


def concatenate_vowels(df):
    dataset = {}  # Dictionary to hold the data for each subject

    # Iterate through each unique subject
    for subject in df["Subject"].unique():
        dataset[subject] = []  # List to hold the numpy arrays for this subject

        # Since we know there are 8 occurrences for each vowel, we loop 8 times
        for i in range(8):
            # Initialize a list to hold the concatenated arrays for this iteration
            concatenated_arrays = []

            # Iterate through each vowel and concatenate the ith occurrence
            for vowel in ["a vowel", "e vowel", "n vowel", "u vowel"]:
                # Filter the DataFrame for the current subject and vowel, and get the ith occurrence
                row_data = df[(df["Subject"] == subject) & (df["Vowel"] == vowel)].iloc[
                    i, :1024
                ]

                # Append the row data to the concatenated arrays list
                concatenated_arrays.append(row_data.values)

            if i == 6 or i == 7:
                # Concatenate along the first axis to get a single array for this subject and iteration
                dataset[subject].append(
                    np.concatenate(concatenated_arrays).astype(np.float32)
                )

    return dataset


def concatenate_vowels_v2(df):
    dataset = {}  # Dictionary to hold the data for each subject

    vowels = ["a vowel", "e vowel", "n vowel", "u vowel"]

    # Generate all possible vowel pairs
    vowel_comb = list(itertools.permutations(vowels, 4))

    # Iterate through each unique subject
    for subject in df["Subject"].unique():
        dataset[subject] = []  # List to hold the numpy arrays for this subject

        # Since we know there are 8 occurrences for each vowel, we loop 8 times
        for i in range(8):
            # Iterate through each vowel and concatenate the ith occurrence
            for vowel_list in vowel_comb:
                # Initialize a list to hold the concatenated arrays for this iteration
                concatenated_arrays = []
                for vowel in vowel_list:
                    # Filter the DataFrame for the current subject and vowel, and get the ith occurrence
                    row_data = df[
                        (df["Subject"] == subject) & (df["Vowel"] == vowel)
                    ].iloc[i, :1024]

                    # Append the row data to the concatenated arrays list
                    concatenated_arrays.append(row_data.values)

                if i == 6 or i == 7:
                    # Concatenate along the first axis to get a single array for this subject and iteration
                    dataset[subject].append(
                        np.concatenate(concatenated_arrays).astype(np.float32)
                    )

    return dataset


# Function to preprocess a signal
def preprocess_signal(input_signal, sampling_rate=9606):
    # Detrend (data in PKL file was already detrended)
    signal_detrended = input_signal  # signal.detrend(input_signal)

    # Remove DC offset
    signal_zero_mean = signal_detrended - np.mean(signal_detrended)

    # Apply Hamming window
    hamming_window = signal.windows.hamming(len(signal_zero_mean))
    signal_windowed = signal_zero_mean * hamming_window

    # # Normalize the signal to have a maximum value of 1
    # signal_normalized = signal_windowed / np.max(np.abs(signal_windowed))

    # Zero Padding
    original_length = len(input_signal)
    fft_length = int(sampling_rate / 1)

    # Calculate the padding length
    padding_length = fft_length - original_length
    if padding_length < 0:
        padding_length = 0  # No padding needed if fft_length is shorter than the signal

    signal_padded = np.pad(signal_windowed, (0, padding_length), "constant")

    return signal_windowed, signal_padded


# Define a function to calculate the amplitude spectrum
def calculate_amplitude_spectrum(s, sampling_rate, cutoff=-1):
    # Perform the Fourier Transform
    fft_result = fft(s)

    # Normalize the FFT result
    fft_normalized = fft_result / len(s)

    # Calculate the amplitude spectrum
    amplitude_spectrum = np.abs(fft_normalized)

    # Only take the first half of the spectrum (positive frequencies)
    half_spectrum = amplitude_spectrum[: len(amplitude_spectrum) // 2]

    # Create a frequency vector
    freq_vector = np.linspace(0, sampling_rate / 2, len(half_spectrum))

    return freq_vector, half_spectrum[:cutoff]


def compute_spectogram_img(input_signal, window_size, overlap):
    fs = len(input_signal) / 0.4264

    # Calculate the STFT and the spectrogram
    frequencies, times, Sxx = signal.spectrogram(
        input_signal, fs=fs, window="hamming", nperseg=window_size, noverlap=overlap
    )

    # Convert the spectrogram to dB
    Sxx_dB = 10 * np.log10(Sxx)

    return frequencies, times, Sxx_dB


def compute_spectogram_npy(input_signal, window_size, overlap):
    fs = len(input_signal) / 0.4264

    # Calculate the STFT and the spectrogram
    frequencies, times, Sxx = signal.spectrogram(
        input_signal, fs=fs, window="hamming", nperseg=window_size, noverlap=overlap
    )

    # Convert the spectrogram to dB
    Sxx_dB = 10 * np.log10(Sxx)

    # Find the index of the frequency that is just above 1300 Hz
    idx = np.where(frequencies <= 1300)[0][-1]

    # Slice the Sxx_dB array to include only the frequencies up to 1300 Hz
    Sxx_dB_limited = Sxx_dB[: idx + 1, :]

    # Normalize the Sxx_dB_limited values to be between 0 and 1
    # Sxx_normalized = (Sxx_dB_limited - np.min(Sxx_dB_limited)) / (
    #     np.max(Sxx_dB_limited) - np.min(Sxx_dB_limited)
    # )

    return Sxx_dB_limited


def save_subject_arrays(efr_data, dataset_name):
    # Create a directory for the dataset if it doesn't exist
    if not os.path.exists(dataset_name):
        os.makedirs(dataset_name)

    spectogram_map = {
        256: [8, 64, 128, 250],
        512: [0, 256, 511],
    }

    for subject_id, signals in efr_data.items():
        for i, input_signal in enumerate(signals):
            # Define the filename with the dataset name, subject, and iteration
            aenu_filename = f"{dataset_name}/{subject_id}_aenu_{i}.npy"
            # print(aenu_filename)
            print("input_signal.shape", input_signal.shape)
            np.save(aenu_filename, input_signal)

            preprocessed_filename = f"{dataset_name}/{subject_id}_preprocessed_{i}.npy"
            # print(aenu_filename)
            preprocessed_signal, preprocessed_signal_padded = preprocess_signal(
                input_signal, sampling_rate=9606
            )
            print("preprocessed_signal.shape", preprocessed_signal.shape)
            np.save(
                preprocessed_filename,
                preprocessed_signal,
            )

            preprocessed_padded_filename = (
                f"{dataset_name}/{subject_id}_preprocessed_padded_{i}.npy"
            )
            # print(aenu_filename)
            print("preprocessed_signal_padded.shape", preprocessed_signal_padded.shape)
            np.save(
                preprocessed_padded_filename,
                preprocessed_signal_padded,
            )

            ampspectra_filename = f"{dataset_name}/{subject_id}_ampspectra_{i}.npy"
            # print(ampspectra_filename)
            ampspectra = calculate_amplitude_spectrum(
                preprocessed_signal_padded, sampling_rate=9606, cutoff=1300
            )[1]
            print("ampspectra.shape", ampspectra.shape)
            np.save(ampspectra_filename, ampspectra)

            for window_size, overlaps in spectogram_map.items():
                for overlap in overlaps:
                    frequencies, times, Sxx_dB = compute_spectogram_img(
                        preprocessed_signal, window_size, overlap
                    )
                    spectogram_filename = f"{dataset_name}/{subject_id}_spectogram_{i}_{window_size}_{overlap}.png"
                    # print(spectogram_filename)
                    # np.save(spectogram_filename, spectogram)

                    # Desired pixel size
                    pixel_size = 32
                    # Choose a DPI (could be any value, but higher DPI means higher resolution)
                    dpi = 512
                    # Calculate the figsize in inches
                    figsize_inch = pixel_size / dpi
                    fig, ax = plt.subplots(figsize=(figsize_inch, figsize_inch))
                    plt.pcolormesh(
                        times, frequencies, Sxx_dB, shading="nearest"
                    )  # Using 'nearest' for a discrete look
                    plt.ylim(0, 1300)
                    plt.axis("off")

                    # Remove padding and margins around the plot
                    plt.margins(0, 0)
                    ax.set_frame_on(False)

                    # Adjust the layout
                    plt.tight_layout(pad=0)

                    # To save the figure without white space
                    fig.savefig(
                        spectogram_filename, bbox_inches="tight", pad_inches=0, dpi=dpi
                    )

                    np.save(
                        f"{dataset_name}/{subject_id}_spectogram_{i}_{window_size}_{overlap}.npy",
                        compute_spectogram_npy(
                            preprocessed_signal, window_size, overlap
                        ),
                    )


df = pd.read_pickle("study2DataFrame.pkl")

all_subjects = df["Subject"].unique()

# Remove rows where 'Avg_Type' is 'EFR'
# df_filtered = df[df['Avg_Type'] != 'EFR']
df_filtered = df[df["Avg_Type"] != "FFR"]

# Split the DataFrame based on the 'Condition' column
df_test = df_filtered[df_filtered["Condition"] == "test"]
df_retest = df_filtered[df_filtered["Condition"] == "retest"]

test_dataset = concatenate_vowels(df_test)
retest_dataset = concatenate_vowels(df_retest)

# You would call the function like this:
save_subject_arrays(test_dataset, "test")
save_subject_arrays(retest_dataset, "retest")


# test_dataset_v2 = concatenate_vowels_v2(df_test)
# retest_dataset_v2 = concatenate_vowels_v2(df_retest)

# # You would call the function like this:
# save_subject_arrays(test_dataset_v2, "test_v2")
# save_subject_arrays(retest_dataset_v2, "retest_v2")
