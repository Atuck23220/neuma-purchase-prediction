# -*- coding: utf-8 -*-
# features_neuma.py
# Loads the NeuMa files and computes the
# per-product features from Section III-D, with the Buy/NoBuy labels.
#   E   = beta / (alpha + theta), averaged over the frontal channels   (7)
#   FAA = ln(alpha at F4) - ln(alpha at F3)                            (8)
#   viewing time = total seconds the product was looked at
# Saves one row per viewed product to features.csv.
#
# Expects the NeuMa .mat files in a folder named neuma_dataset/ next to
# this script.

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import welch

data_folder = "neuma_dataset"

# ---- Settings ----
min_view_seconds = 1.0     # EEG features are only computed at or above this
bands = {"theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}   # Hz
frontal = ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz"]

os.makedirs("maps", exist_ok=True)

# Power in each band for every channel of one product's EEG.
# Welch's method cuts the signal into 1 second windows, takes the power
# spectrum of each, and averages them. A 1 second window at 300 Hz gives
# one spectrum value per 1 Hz, so a band's power is the sum of its values.
def band_powers(eeg, fs):
    window = int(fs)                                   # 300 samples = 1 s
    freqs, psd = welch(eeg, fs=fs, nperseg=window)     # psd is [channels x freqs]
    powers = {}
    for band_name, (low, high) in bands.items():
        in_band = (freqs >= low) & (freqs < high)
        powers[band_name] = psd[:, in_band].sum(axis=1)
    return powers

# Reads one participant file and returns one row per viewed product:
# [subject number, page, product, bought, viewing seconds, E, FAA]
# E and FAA are left as nan when the product was viewed for less than
# min_view_seconds, since there is too little signal to measure them.
def subject_features(path):
    name = os.path.basename(path).replace(".mat", "")
    subject = loadmat(path, squeeze_me=True, struct_as_record=False)[name]
    eeg_all = subject.EEG_clean.Data                   # [19 channels x samples]
    fs = subject.EEG_clean.Fs

    # Find which row of the EEG belongs to each channel name.
    labels = [channel.labels for channel in subject.EEG_clean.chans]
    frontal_rows = [labels.index(channel) for channel in frontal]
    f3 = labels.index("F3")
    f4 = labels.index("F4")

    rows = []
    for page in range(1, 7):
        page_data = getattr(subject, f"Page{page}")
        for product in range(1, 25):
            product_data = getattr(page_data, f"Product{product}")
            bought = int(product_data.ProductInfo.Bought)

            segments = np.array(product_data.EEG_segments)
            if segments.size == 0:
                continue                               # never looked at
            segments = np.atleast_2d(segments)

            # Concatenate this product's EEG segments. Matlab counts from 1
            # and includes both ends, so start - 1 up to end in Python.
            pieces = [eeg_all[:, start - 1:end] for start, end in segments]
            eeg = np.hstack(pieces)
            seconds = eeg.shape[1] / fs

            if seconds >= min_view_seconds:
                p = band_powers(eeg, fs)
                engagement = p["beta"] / (p["alpha"] + p["theta"])
                E = engagement[frontal_rows].mean()
                FAA = np.log(p["alpha"][f4]) - np.log(p["alpha"][f3])
            else:
                E = np.nan
                FAA = np.nan

            rows.append([int(name[1:]), page, product, bought, seconds, E, FAA])
    return name, np.array(rows)

# ---- Run every participant ----
files = sorted(glob.glob(f"{data_folder}/S*.mat"))
print(f"Found {len(files)} participant files in {data_folder}/")
print(f"{'subj':<6}{'viewed':>8}{'with EEG':>10}{'Buy':>6}{'NoBuy':>7}{'mean E':>9}{'mean FAA':>10}")

all_rows = []
for path in files:
    name, rows = subject_features(path)
    all_rows.append(rows)

    has_eeg = ~np.isnan(rows[:, 5])
    buy = (rows[:, 3] == 1) & has_eeg
    nobuy = (rows[:, 3] == 0) & has_eeg
    print(f"{name:<6}{len(rows):>8}{has_eeg.sum():>10}{buy.sum():>6}{nobuy.sum():>7}"
          f"{rows[has_eeg, 5].mean():>9.3f}{rows[has_eeg, 6].mean():>10.3f}")

features = np.vstack(all_rows)

# ---- Save ----
np.savetxt("features.csv", features, delimiter=",",
           header="subject,page,product,bought,view_seconds,E,FAA",
           comments="", fmt=["%d", "%d", "%d", "%d", "%.4f", "%.6f", "%.6f"])

has_eeg = ~np.isnan(features[:, 5])
buy = (features[:, 3] == 1) & has_eeg
nobuy = (features[:, 3] == 0) & has_eeg
print()
print(f"Saved {len(features)} rows to features.csv")
print(f"Rows with EEG features (viewed >= {min_view_seconds} s): {has_eeg.sum()} "
      f"({buy.sum()} Buy, {nobuy.sum()} NoBuy)")
print(f"Mean E    Buy {features[buy, 5].mean():.4f}   NoBuy {features[nobuy, 5].mean():.4f}")
print(f"Mean FAA  Buy {features[buy, 6].mean():.4f}   NoBuy {features[nobuy, 6].mean():.4f}")

# ---- First look: do the two EEG features separate Buy from NoBuy? ----
# Raw values from all participants on one plot, before any normalization.
plt.figure(figsize=(6, 5))
plt.scatter(features[nobuy, 5], features[nobuy, 6], s=5, alpha=0.4, label="NoBuy (0)")
plt.scatter(features[buy, 5], features[buy, 6], s=5, alpha=0.7, label="Buy (1)")
plt.xlabel("engagement index E")
plt.ylabel("frontal alpha asymmetry FAA")
plt.title(f"EEG features, all participants (viewed >= {min_view_seconds} s)")
plt.legend()
plt.savefig("maps/features_E_vs_FAA.png", dpi=150, bbox_inches="tight")
plt.show()
