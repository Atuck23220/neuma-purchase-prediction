# -*- coding: utf-8 -*-
# figures_neuma.py
# Makes the paper figures from files that are already saved. No training
# happens here, so it runs in seconds and can be rerun any time.
#   features.csv                   -> viewing-time histogram, E vs FAA scatter
#   per_participant_within.csv     -> box plot of balanced accuracy
#   per_participant_loso.csv          per participant
#   stage1b_TE_history.csv         -> total error per training cycle
# The figures use the same look as the plots from the experiment scripts:
# matplotlib's standard blue and orange, a title on top, and a legend box.

import os
import numpy as np
import matplotlib.pyplot as plt

min_view_seconds = 1.0
loso_cycle_limit = 500        # the leave-one-subject-out cycle limit
blue = "tab:blue"             # matplotlib's standard first two colors
orange = "tab:orange"

os.makedirs("maps", exist_ok=True)

def save(name):
    plt.savefig(f"maps/{name}.png", dpi=150, bbox_inches="tight")
    plt.show()

# ---- Load the features ----
# Columns: 0 subject, 1 page, 2 product, 3 bought, 4 view_seconds, 5 E, 6 FAA
features = np.genfromtxt("features.csv", delimiter=",", skip_header=1)
bought = features[:, 3] == 1
seconds = features[:, 4]
has_eeg = seconds >= min_view_seconds

# ---- Figure: viewing time per product, Buy vs NoBuy ----
# Log-spaced bins, because most views are short and a few are long. The
# dashed line is the 1.0 s minimum.
bins = np.logspace(np.log10(0.01), np.log10(seconds.max()), 40)
plt.figure(figsize=(7, 4.5))
plt.hist(seconds[~bought], bins=bins, color=blue, alpha=0.6, label="NoBuy (0)")
plt.hist(seconds[bought], bins=bins, color=orange, alpha=0.6, label="Buy (1)")
plt.axvline(min_view_seconds, color="k", linestyle="--", linewidth=0.8, label="1.0 s minimum")
plt.xscale("log")
plt.xlabel("viewing time per product (seconds, log scale)")
plt.ylabel("number of products")
plt.title("Viewing time per product, all participants")
plt.legend()
save("fig_viewing_time")

# ---- Figure: the two EEG features, Buy vs NoBuy ----
# One point per product viewed at least 1.0 s. NoBuy is drawn first as
# blue, Buy on top in orange.
plt.figure(figsize=(6, 5))
plt.scatter(features[has_eeg & ~bought, 5], features[has_eeg & ~bought, 6],
            s=5, color=blue, alpha=0.4, label="NoBuy (0)")
plt.scatter(features[has_eeg & bought, 5], features[has_eeg & bought, 6],
            s=5, color=orange, alpha=0.7, label="Buy (1)")
plt.xlabel("engagement index E")
plt.ylabel("frontal alpha asymmetry FAA")
plt.title(f"EEG features, all participants (viewed >= {min_view_seconds} s)")
plt.legend()
save("fig_eeg_features")

# ---- Load the per-participant results ----
# Each line: test, input_set, activation, class_weights, subject, TP, FN, FP, TN
def read_counts(filename):
    rows = []
    with open(filename) as file:
        next(file)                                  # skip the header line
        for line in file:
            parts = line.strip().split(",")
            rows.append(parts[:4] + [int(value) for value in parts[4:]])
    return rows

results = read_counts("per_participant_within.csv") + read_counts("per_participant_loso.csv")

# Balanced accuracy for each participant in one experiment:
# the average of the true positive rate and the true negative rate.
def balanced_per_participant(test, input_set, activation, class_weights):
    values = []
    for row in results:
        if row[:4] == [test, input_set, activation, class_weights]:
            subject, TP, FN, FP, TN = row[4:]
            values.append((TP / (TP + FN) + TN / (FP + TN)) / 2)
    return np.array(values)

# ---- Figure: balanced accuracy per participant ----
# Soft activation with class weights on. Each box holds the 42
# participants: the line in the box is the median, the box covers the
# middle half, and the whiskers reach the lowest and highest participant.
# The dashed line at 0.5 is what guessing would score.
input_sets = ["time only", "EEG only", "EEG + time"]
set_labels = ["viewing time only", "EEG only", "EEG + viewing time"]

plt.figure(figsize=(7, 4.5))
print(f"{'test':<8}{'input set':<12}{'lowest':>8}{'quarter':>9}{'median':>8}{'3 quarters':>12}{'highest':>9}")
for group, (test, color, label) in enumerate((("within", blue, "within-subject"),
                                              ("loso", orange, "leave-one-subject-out"))):
    for position, input_set in enumerate(input_sets):
        values = balanced_per_participant(test, input_set, "soft", "on")
        x = position + (group - 0.5) * 0.36
        box = plt.boxplot(values, positions=[x], widths=0.3, whis=(0, 100), patch_artist=True,
                          medianprops={"color": "black"},
                          boxprops={"facecolor": color, "alpha": 0.7})
        if position == 0:
            box["boxes"][0].set_label(label)
        low, q1, median, q3, high = np.percentile(values, [0, 25, 50, 75, 100])
        print(f"{test:<8}{input_set:<12}{low:>8.3f}{q1:>9.3f}{median:>8.3f}{q3:>12.3f}{high:>9.3f}")

plt.axhline(0.5, color="k", linestyle="--", linewidth=0.8, label="chance (0.5)")
plt.xticks(range(len(input_sets)), set_labels)
plt.xlim(-0.5, 2.5)
plt.ylim(0.2, 1.0)
plt.ylabel("balanced accuracy per participant")
plt.title("Balanced accuracy per participant (soft activation, class weights on)")
plt.legend(loc="lower right")
save("fig_balanced_accuracy")

# ---- Figure: total error per training cycle ----
# One leave-one-subject-out training run for each activation (saved by
# Stage 1b of Paper_Neuma.py). The soft curve flattens out well before the
# 500-cycle limit. The hard curve never settles.
history = np.genfromtxt("stage1b_TE_history.csv", delimiter=",", skip_header=1)
plt.figure(figsize=(7, 4.5))
plt.plot(history[:, 0], history[:, 1], color=blue, label="hard unipolar")
plt.plot(history[:, 0], history[:, 2], color=orange, label="soft unipolar")
plt.axvline(loso_cycle_limit, color="k", linestyle="--", linewidth=0.8,
            label=f"{loso_cycle_limit}-cycle limit")
plt.xscale("log")
plt.xlabel("training cycle (log scale)")
plt.ylabel("total error TE (class weighted)")
plt.title("Total error per cycle, leave-one-subject-out (S01 held out)")
plt.legend()
save("fig_error_per_cycle")