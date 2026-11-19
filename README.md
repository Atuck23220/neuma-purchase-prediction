# EEG and Eye-Tracking Purchase Prediction (NeuMa)

Code for a research project testing whether EEG recorded while people browse products predicts what they choose to buy, beyond how long they look at each product. Uses the public NeuMa neuromarketing dataset (42 participants, EEG plus eye tracking).

**Status:** paper in preparation. Results are not included in this repository until the paper is submitted.

## Team

Aaron Tuck, Abigiya Yohannes, Ahmad Abubakr Siddiq, Akshita Ramesh
Virginia Commonwealth University. An independent team research paper, written alongside CMSC 436 and not a course requirement.

All code in this repository was written by Aaron Tuck.

## What the code does

| Script | Purpose |
|---|---|
| `features_neuma.py` | Loads the NeuMa participant files and computes per-product features: an EEG engagement index, frontal alpha asymmetry, and eye-tracking viewing time. Writes `features.csv`. |
| `Paper_Neuma.py` | Trains a single perceptron, implemented from scratch, to predict Buy or NoBuy for each product, and runs the evaluation. |
| `figures_neuma.py` | Draws the paper figures from saved results. No training, so it reruns in seconds. |

### Features

- EEG band power (theta, alpha, beta) per channel with Welch's method, 1 s Hann windows, 50% overlap
- Engagement index: beta / (alpha + theta), averaged over the frontal channels Fp1, Fp2, F3, F4, F7, F8, Fz
- Frontal alpha asymmetry: ln(alpha at F4) minus ln(alpha at F3)
- Viewing time: total seconds the product was looked at, from the eye tracker
- EEG features are computed only for products viewed at least 1 s

### Model

- Single perceptron with hard and soft unipolar activation, trained with the per-pattern perceptron learning rule
- Optional class weighting so the smaller Buy class counts as much as NoBuy
- Patterns reshuffled every training cycle so no fixed order is built into the weights
- Fixed random seed so folds, starting weights, and shuffles repeat exactly

### Evaluation design

- Three input sets: viewing time only, EEG only, and EEG plus viewing time
- Within-subject: stratified 5-fold cross-validation inside each participant, with the same folds for every experiment
- Leave-one-subject-out: train on 41 participants, test on the one held out
- Min-max normalization fit on training data only, so nothing about the test data leaks into training
- Full confusion matrix and balanced accuracy, since the classes are imbalanced

## Requirements

Python 3.13 with NumPy, SciPy, and Matplotlib. The versions used for the paper are pinned in `requirements.txt`. The random seed is fixed, so the same versions give the same numbers.

```
pip install -r requirements.txt
```

## Data

The NeuMa dataset is not included. Download the preprocessed release, [NeuMa_PreProcessed on figshare](https://figshare.com/articles/dataset/NeuMa_PreProcessed_A_multimodal_Neuromarketing_dataset/22117124/3), and place the participant files (`S01.mat`, `S02.mat`, ...) in a folder named `neuma_dataset/` next to the scripts.

## How to run

```
python features_neuma.py     # writes features.csv
python Paper_Neuma.py        # runs the stages switched on at the top of the file
python figures_neuma.py      # draws the figures into maps/
```

`Paper_Neuma.py` runs in stages, each with an on/off switch near the top of the file:

| Switch | Stage |
|---|---|
| `run_stage1` | Timed within-subject test on a few participants |
| `run_stage1b` | Total error per cycle for one leave-one-subject-out fold |
| `run_within` | Full within-subject experiments |
| `run_loso` | Full leave-one-subject-out experiments |
| `run_extra` | Viewing time only on every viewed product |
| `run_sweep` | Tuning sweep, with settings chosen from training scores only |

`figures_neuma.py` needs `features.csv`, `per_participant_within.csv`, `per_participant_loso.csv`, and `stage1b_TE_history.csv`, which the first two scripts write.
