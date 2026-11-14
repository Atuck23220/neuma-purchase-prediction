# -*- coding: utf-8 -*-
# Paper_Neuma.py
# Aaron Tuck, Abigiya Yohannes, Ahmad Abubakr Siddiq, Akshita Ramesh
# Perceptron experiments for the NeuMa purchase prediction paper.
# Reads features.csv (made by features_neuma.py), trains a perceptron with
# hard and soft unipolar activation to predict Buy (1) or NoBuy (0), and
# reports the confusion matrix and accuracy measures.
#
# The file runs in stages, each with its own on/off switch below:
#   Stage 1   a small timed within-subject test (how long does training take)
#   Stage 1b  total error per cycle for one leave-one-subject-out fold
#             (the evidence for the 500-cycle limit in that test)
#   Stage 2   the full experiments:
#             within-subject and leave-one-subject-out
#             x 3 input sets x hard/soft x class weights on/off,
#             plus viewing time only on every viewed product
#   Stage 3   tuning sweep: the learning constant and gain are changed one
#             at a time, and each setting is scored on the training data
#             and on the test data

import os
import math
import time
import numpy as np
import matplotlib.pyplot as plt

# ---- Settings from the paper ----
max_iteration = 5000        # max training cycles, within-subject
max_iteration_loso = 500    # max training cycles, leave-one-subject-out
min_view_seconds = 1.0      # only products viewed at least this long
n_folds = 5                 # within-subject cross-validation folds
seed = 436                  # fixed so folds and starting weights repeat

# ---- Tuning (set experimentally) ----
# alpha_soft was 0.1 at first. With steps that large, the final weights
# depended on the order the products were shown in. At 0.001 the results
# are the same from one shuffle to the next.
alpha_hard = 0.001          # learning constant, hard
alpha_soft = 0.001          # learning constant, soft
k = 5                       # gain, soft only
epsilon_per_product = 0.05  # TE target = this x number of training products

# ---- Which stages to run ----
run_stage1 = False          # timed within-subject test (already done)
run_stage1b = True          # error per cycle, one fold
run_within = False          # Stage 2: full within-subject experiments (already done)
run_loso = False            # Stage 2: full leave-one-subject-out experiments (already done)
run_extra = False           # Stage 2: viewing time only, all viewed products (already done)
run_sweep = False           # Stage 3: tuning sweep

# ---- Stage 1 and 1b settings ----
stage1_subjects = 5                 # how many participants to time
stage1_input_set = "EEG + time"     # the largest input set, so the slowest
use_class_weights = True
stage1b_holdout = 1                 # participant left out for testing (S01)
checkpoints = [1, 10, 50, 100, 200, 500, 1000, 2000, 3000, 4000, 5000]

# ---- Stage 3 settings ----
# One setting is changed at a time around the starting values above.
sweep_tests = ["within", "loso"]
sweep_input_set = "EEG + time"
sweep_alpha_hard = [0.0001, 0.001, 0.01]
sweep_soft = [(0.1, 5), (0.01, 5), (0.001, 5), (0.001, 1), (0.001, 10)]   # (alpha, k)

# Columns of features.csv:
# 0 subject, 1 page, 2 product, 3 bought, 4 view_seconds, 5 E, 6 FAA
input_sets = {"time only": [4], "EEG only": [5, 6], "EEG + time": [4, 5, 6]}
column_names = {4: "time", 5: "E", 6: "FAA"}

# ---- Results files ----
# Everything printed also goes to a text file. Each stage starts its own,
# and each line is saved right away, so a run that stops partway keeps
# everything it finished.
os.makedirs("maps", exist_ok=True)
os.makedirs("train_test_data", exist_ok=True)
results_file = None

def start_results(name):
    global results_file
    results_file = open(name, "w")

def report(text=""):
    print(text)
    results_file.write(text + "\n")
    results_file.flush()

# Min-max normalization. The min and max come from the training data only,
# then the same numbers are used on the test data, so nothing about the
# test products leaks into training. Test values can land a little outside
# [0, 1], which is fine.
def normalize_fit(train_inputs):
    low = train_inputs.min(axis=0)
    high = train_inputs.max(axis=0)
    spread = high - low
    spread[spread == 0] = 1          # avoids dividing by zero
    return low, spread

def normalize_apply(inputs, low, spread):
    return (inputs - low) / spread

# Hard unipolar: 1 if net >= 0, else 0
def hard_unipolar(net):
    if net >= 0:
        return 1
    else:
        return 0

# Soft unipolar: S-curve between 0 and 1, k is the gain
# The two checks only matter for huge values of k * net, where exp() would
# overflow. The output there is already 0 or 1, so nothing changes.
def soft_unipolar(net, k):
    z = -k * net
    if z > 500:
        return 0.0
    if z < -500:
        return 1.0
    return 1 / (1 + math.exp(z))

# Splits one participant's products into folds, keeping Buy and NoBuy
# spread evenly so every fold has some of each (stratified). Returns a list
# with one array of row numbers per fold.
def stratified_folds(labels, n_folds):
    buy_rows = np.random.permutation(np.where(labels == 1)[0])
    nobuy_rows = np.random.permutation(np.where(labels == 0)[0])
    folds = []
    for fold in range(n_folds):
        # every n_folds-th Buy row and NoBuy row goes to this fold
        rows = np.concatenate((buy_rows[fold::n_folds], nobuy_rows[fold::n_folds]))
        folds.append(rows)
    return folds

# Trains one neuron (hard or soft activation) on normalized inputs.
# The last weight is the bias: it is multiplied by a constant input of 1.
# With class weights on, a Buy product's update and its error term are
# multiplied by (number of NoBuy / number of Buy) in the training data, so
# the few Buy products count as much as the many NoBuy ones.
#
# The products are shown in a new random order every cycle. The weights
# change after every product, so they lean toward whichever products came
# last. If the order never changed (for example, always ending on the same
# participant), that lean would be built into the results.
#
# The inner loop uses plain Python lists instead of NumPy, because NumPy is
# slow when it handles one product at a time. The math is the same.
#
# It also keeps the total error of every cycle (TE_history) and a copy of
# the weights at each checkpoint cycle, so training can be looked at
# afterwards without running it again.
def train_perceptron(inputs, labels, activation, alpha, epsilon, max_cycles, k=1,
                     class_weights=True, checkpoints=()):
    n_inputs = inputs.shape[1]
    patterns = inputs.tolist()
    targets = labels.tolist()

    n_buy = np.sum(labels == 1)
    n_nobuy = np.sum(labels == 0)
    pattern_weight = np.ones(len(labels))
    if class_weights:
        pattern_weight[labels == 1] = n_nobuy / n_buy
    pattern_weight = pattern_weight.tolist()

    w = np.random.uniform(-0.5, 0.5, n_inputs + 1).tolist()
    TE_history = []
    saved_weights = {}

    for iteration in range(max_cycles):
        order = np.random.permutation(len(patterns)).tolist()   # new order every cycle
        TE = 0
        for pattern in order:
            x = patterns[pattern]
            net = w[n_inputs]                            # bias weight x constant input 1
            for i in range(n_inputs):
                net = net + w[i] * x[i]
            if activation == "hard":
                out = hard_unipolar(net)
            else:
                out = soft_unipolar(net, k)
            err = targets[pattern] - out
            learn = alpha * err * pattern_weight[pattern]
            for i in range(n_inputs):
                w[i] = w[i] + learn * x[i]
            w[n_inputs] = w[n_inputs] + learn
            TE = TE + pattern_weight[pattern] * err**2
        TE_history.append(TE)
        if (iteration + 1) in checkpoints:
            saved_weights[iteration + 1] = np.array(w)
        if TE < epsilon:
            break

    # +1 because iteration starts at 0
    return np.array(w), TE, iteration + 1, TE_history, saved_weights

# Tests fixed weights on a test set and returns the four counts.
# net >= 0 works for both hard and soft, since soft is 0.5 at net = 0.
def count_confusion(inputs, labels, w):
    ones = np.ones((len(inputs), 1))
    inputs_aug = np.hstack((inputs, ones))
    net = np.dot(inputs_aug, w)
    predicted = (net >= 0).astype(int)

    truePos = np.sum((predicted == 1) & (labels == 1))
    falseNeg = np.sum((predicted == 0) & (labels == 1))
    falsePos = np.sum((predicted == 1) & (labels == 0))
    trueNeg = np.sum((predicted == 0) & (labels == 0))
    return np.array([truePos, falseNeg, falsePos, trueNeg])

# A rate as text, or "n/a" when its denominator is zero (for example,
# precision when nothing was predicted Buy).
def rate(top, bottom):
    if bottom == 0:
        return "   n/a"
    return f"{top / bottom:.4f}"

# Prints the confusion matrix and accuracy measures from the four counts.
# Counts from many folds are added up first, then the rates are computed
# from the totals.
def report_confusion(heading, counts):
    truePos, falseNeg, falsePos, trueNeg = [int(c) for c in counts]
    total = truePos + falseNeg + falsePos + trueNeg
    actualBuy = truePos + falseNeg
    actualNoBuy = falsePos + trueNeg
    predBuy = truePos + falsePos
    predNoBuy = falseNeg + trueNeg

    report(heading)
    report("              Predicted 1   Predicted 0        Total")
    report(f"  Actual 1    {truePos:9d}     {falseNeg:9d}    {actualBuy:9d}")
    report(f"  Actual 0    {falsePos:9d}     {trueNeg:9d}    {actualNoBuy:9d}")
    report(f"  Total       {predBuy:9d}     {predNoBuy:9d}    {total:9d}")
    report(f"  accuracy            {rate(truePos + trueNeg, total)}")
    report(f"  error               {rate(falsePos + falseNeg, total)}")
    report(f"  true positive rate  {rate(truePos, actualBuy)}")
    report(f"  true negative rate  {rate(trueNeg, actualNoBuy)}")
    report(f"  false positive rate {rate(falsePos, actualNoBuy)}")
    report(f"  false negative rate {rate(falseNeg, actualBuy)}")
    report(f"  precision           {rate(truePos, predBuy)}")
    report(f"  always-NoBuy accuracy would be {rate(actualNoBuy, total)}")

# ---- Load the features ----
# features_all has every viewed product. features keeps only the ones
# viewed for at least min_view_seconds, which are the ones with EEG features.
features_all = np.genfromtxt("features.csv", delimiter=",", skip_header=1)
features = features_all[features_all[:, 4] >= min_view_seconds]
subjects = np.unique(features[:, 0]).astype(int)

# The within-subject folds are made once here and reused by every Stage 2
# experiment, so all input sets and activations are tested on exactly the
# same splits. (The order of the products inside a fold does not matter,
# because training reshuffles them every cycle.)
np.random.seed(seed)
folds_main = {}
folds_all = {}
for subject in subjects:
    folds_main[subject] = stratified_folds(features[features[:, 0] == subject, 3], n_folds)
    folds_all[subject] = stratified_folds(features_all[features_all[:, 0] == subject, 3], n_folds)

def report_settings():
    report(f"Settings: alpha_hard = {alpha_hard}, alpha_soft = {alpha_soft}, k = {k}, "
           f"seed = {seed}, max_iteration = {max_iteration} (within-subject), "
           f"{max_iteration_loso} (leave-one-subject-out)")
    report(f"epsilon = {epsilon_per_product} x training products, min view = {min_view_seconds} s")
    report(f"Loaded {len(features)} products from {len(subjects)} participants "
           f"({int(np.sum(features[:, 3] == 1))} Buy, {int(np.sum(features[:, 3] == 0))} NoBuy)")

# ---- Stage 1: timed within-subject test ----
if run_stage1:
    np.random.seed(seed)
    start_results("results_stage1.txt")
    report_settings()
    # For each participant: split their products into 5 folds, train on 4,
    # test on the 1 left out, and repeat so every product is tested once.
    columns = input_sets[stage1_input_set]
    report()
    report(f"Stage 1: within-subject, input set '{stage1_input_set}', "
           f"first {stage1_subjects} participants")

    total_updates = 0
    total_seconds = 0
    for activation in ("hard", "soft"):
        if activation == "hard":
            alpha = alpha_hard
        else:
            alpha = alpha_soft

        report()
        report(f"{activation} activation")
        report(f"{'subj':<6}{'products':>9}{'Buy':>5}{'mean cycles':>13}{'reached target':>16}{'seconds':>9}")

        all_counts = np.zeros(4)
        for subject in subjects[:stage1_subjects]:
            rows = features[features[:, 0] == subject]
            inputs = rows[:, columns]
            labels = rows[:, 3]
            folds = stratified_folds(labels, n_folds)

            cycles_list = []
            reached = 0
            start = time.time()
            for fold in range(n_folds):
                test_rows = folds[fold]
                train_rows = np.concatenate([folds[other] for other in range(n_folds) if other != fold])

                low, spread = normalize_fit(inputs[train_rows])
                train_inputs = normalize_apply(inputs[train_rows], low, spread)
                test_inputs = normalize_apply(inputs[test_rows], low, spread)

                epsilon = epsilon_per_product * len(train_rows)
                w, TE, cycles, TE_history, saved_weights = train_perceptron(
                    train_inputs, labels[train_rows], activation, alpha, epsilon,
                    max_iteration, k, use_class_weights)
                all_counts = all_counts + count_confusion(test_inputs, labels[test_rows], w)

                cycles_list.append(cycles)
                if cycles < max_iteration:
                    reached = reached + 1
                total_updates = total_updates + cycles * len(train_rows)
            seconds = time.time() - start
            total_seconds = total_seconds + seconds

            report(f"S{subject:02d}   {len(rows):>9}{int(np.sum(labels == 1)):>5}"
                   f"{np.mean(cycles_list):>13.0f}{f'{reached} of {n_folds}':>16}{seconds:>9.1f}")

        report()
        report_confusion(f"Within-subject, {stage1_input_set}, {activation}, "
                         f"first {stage1_subjects} participants (all folds added up)", all_counts)

    # Uses the measured speed to estimate the full runs if every run goes to
    # the cycle limit: 3 input sets x 2 activations x weighted/unweighted = 12.
    updates_per_second = total_updates / total_seconds
    within_updates = len(features) * (n_folds - 1) / n_folds * max_iteration * n_folds * 12
    loso_updates = len(features) * (len(subjects) - 1) / len(subjects) * max_iteration * len(subjects) * 12

    report()
    report("Timing")
    report(f"  weight updates done       {total_updates}")
    report(f"  seconds                   {total_seconds:.1f}")
    report(f"  updates per second        {updates_per_second:.0f}")
    report(f"  full within-subject test, worst case   {within_updates / updates_per_second / 60:.0f} minutes")
    report(f"  full leave-one-subject-out, worst case {loso_updates / updates_per_second / 3600:.1f} hours")

    results_file.close()
    print("\nResults saved to results_stage1.txt")

# ---- Stage 1b: error per cycle for one leave-one-subject-out fold ----
# Train on every participant except one, test on the one left out. The
# total error is kept for every cycle, and the held-out participant is
# tested with the weights saved at each checkpoint cycle.
if run_stage1b:
    np.random.seed(seed)
    start_results("results_stage1b.txt")
    report_settings()

    columns = input_sets[stage1_input_set]
    is_test = features[:, 0] == stage1b_holdout
    train_labels = features[~is_test, 3]
    test_labels = features[is_test, 3]

    low, spread = normalize_fit(features[~is_test][:, columns])
    train_inputs = normalize_apply(features[~is_test][:, columns], low, spread)
    test_inputs = normalize_apply(features[is_test][:, columns], low, spread)
    epsilon = epsilon_per_product * len(train_inputs)

    report()
    report(f"Stage 1b: leave-one-subject-out, S{stage1b_holdout:02d} held out, "
           f"input set '{stage1_input_set}'")
    report(f"Training products {len(train_inputs)}, test products {len(test_inputs)} "
           f"({int(np.sum(test_labels == 1))} Buy), TE target {epsilon:.1f}")

    plt.figure(figsize=(7, 4.5))
    histories = {}
    for activation in ("hard", "soft"):
        if activation == "hard":
            alpha = alpha_hard
        else:
            alpha = alpha_soft

        start = time.time()
        w, TE, cycles, TE_history, saved_weights = train_perceptron(
            train_inputs, train_labels, activation, alpha, epsilon, max_iteration, k,
            use_class_weights, checkpoints)
        seconds = time.time() - start
        histories[activation] = TE_history

        report()
        report(f"{activation} activation: {cycles} cycles in {seconds:.0f} seconds, final TE = {TE:.2f}")
        report(f"{'cycle':>7}{'TE':>12}{'accuracy':>10}{'TPR':>8}{'TNR':>8}{'precision':>11}")
        for cycle in checkpoints:
            if cycle not in saved_weights:
                continue                                # training stopped before this cycle
            truePos, falseNeg, falsePos, trueNeg = count_confusion(test_inputs, test_labels,
                                                                  saved_weights[cycle])
            report(f"{cycle:>7}{TE_history[cycle - 1]:>12.2f}"
                   f"{rate(truePos + trueNeg, len(test_labels)):>10}"
                   f"{rate(truePos, truePos + falseNeg):>8}{rate(trueNeg, falsePos + trueNeg):>8}"
                   f"{rate(truePos, truePos + falsePos):>11}")

        plt.plot(range(1, cycles + 1), TE_history, label=f"{activation} unipolar")

    plt.xscale("log")
    plt.xlabel("training cycle (log scale)")
    plt.ylabel("total error TE (class weighted)")
    plt.title(f"Total error per cycle, leave-one-subject-out (S{stage1b_holdout:02d} held out)")
    plt.legend()
    plt.savefig("maps/stage1b_TE_per_cycle.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Save the total error of every cycle, so figures_neuma.py can draw the
    # paper version of this plot without training again. If one activation
    # stopped early, its remaining cycles are left as nan.
    table = np.full((max_iteration, 3), np.nan)
    table[:, 0] = np.arange(1, max_iteration + 1)
    table[:len(histories["hard"]), 1] = histories["hard"]
    table[:len(histories["soft"]), 2] = histories["soft"]
    np.savetxt("stage1b_TE_history.csv", table, delimiter=",", fmt=["%d", "%.4f", "%.4f"],
               header="cycle,hard,soft", comments="")

    results_file.close()
    print("\nResults saved to results_stage1b.txt and stage1b_TE_history.csv")

# =====================================================================
# Stage 2: the full experiments
# =====================================================================

# Runs one experiment: one test (within-subject or leave-one-subject-out)
# with one input set, one activation, and class weights on or off.
# Returns the four confusion counts for every participant, the weights of
# every trained neuron, how many cycles each one took, how many reached
# the error target, and the confusion counts on the training data.
# alpha and gain are normally taken from the settings at the top. The
# tuning sweep passes its own values instead.
def run_experiment(data, folds_by_subject, test, columns, activation, class_weights,
                   alpha=None, gain=None):
    # Same seed for every experiment, so they all get the same sequence of
    # random starting weights and shuffles.
    np.random.seed(seed)
    if alpha is None:
        if activation == "hard":
            alpha = alpha_hard
        else:
            alpha = alpha_soft
    if gain is None:
        gain = k

    counts_by_subject = []
    weight_rows = []          # [subject, fold, w1, ..., bias]
    cycles_list = []
    reached = 0
    train_counts = np.zeros(4)

    for subject in subjects:
        is_subject = data[:, 0] == subject
        subject_counts = np.zeros(4)

        if test == "within":
            # 5 folds inside this participant: train on 4, test on 1.
            inputs = data[is_subject][:, columns]
            labels = data[is_subject, 3]
            folds = folds_by_subject[subject]
            splits = []
            for fold in range(n_folds):
                train_rows = np.concatenate([folds[other] for other in range(n_folds) if other != fold])
                splits.append((inputs[train_rows], labels[train_rows],
                               inputs[folds[fold]], labels[folds[fold]]))
            max_cycles = max_iteration
        else:
            # Train on the other participants, test on this one.
            splits = [(data[~is_subject][:, columns], data[~is_subject, 3],
                       data[is_subject][:, columns], data[is_subject, 3])]
            max_cycles = max_iteration_loso

        for fold, (train_inputs, train_labels, test_inputs, test_labels) in enumerate(splits):
            low, spread = normalize_fit(train_inputs)
            train_inputs = normalize_apply(train_inputs, low, spread)
            test_inputs = normalize_apply(test_inputs, low, spread)

            epsilon = epsilon_per_product * len(train_inputs)
            w, TE, cycles, TE_history, saved_weights = train_perceptron(
                train_inputs, train_labels, activation, alpha, epsilon, max_cycles, gain, class_weights)

            subject_counts = subject_counts + count_confusion(test_inputs, test_labels, w)
            train_counts = train_counts + count_confusion(train_inputs, train_labels, w)
            weight_rows.append([subject, fold + 1] + list(w))
            cycles_list.append(cycles)
            if cycles < max_cycles:
                reached = reached + 1

        counts_by_subject.append(subject_counts)

    return np.array(counts_by_subject), weight_rows, cycles_list, reached, train_counts

# Runs a list of experiments and writes three files named after the tag:
#   results_<tag>.txt           confusion matrix per experiment + summary table
#   per_participant_<tag>.csv   the four counts for each participant
#   weights_<tag>.csv           the weights of every trained neuron
# experiments is a list of (test, input set name, activation, class weights).
def run_block(tag, title, data, folds_by_subject, experiments):
    start_results(f"results_{tag}.txt")
    report_settings()
    report()
    report(f"{title}: {len(data)} products "
           f"({int(np.sum(data[:, 3] == 1))} Buy, {int(np.sum(data[:, 3] == 0))} NoBuy)")

    participant_file = open(f"per_participant_{tag}.csv", "w")
    participant_file.write("test,input_set,activation,class_weights,subject,TP,FN,FP,TN\n")
    weights_file = open(f"weights_{tag}.csv", "w")
    weights_file.write("test,input_set,activation,class_weights,subject,fold,weights (inputs in order, bias last)\n")

    summary = []
    for test, set_name, activation, class_weights in experiments:
        columns = input_sets[set_name]
        if class_weights:
            weight_text = "on"
        else:
            weight_text = "off"
        label = f"{test} | {set_name} | {activation} | class weights {weight_text}"

        start = time.time()
        counts_by_subject, weight_rows, cycles_list, reached, train_counts = run_experiment(
            data, folds_by_subject, test, columns, activation, class_weights)
        seconds = time.time() - start
        counts = counts_by_subject.sum(axis=0)

        report()
        report_confusion(label, counts)

        # Average weight for each input across all the trained neurons.
        # Inputs are normalized, so the sizes can be compared with each other.
        all_weights = np.array([row[2:] for row in weight_rows])
        names = [column_names[c] for c in columns] + ["bias"]
        weight_summary = ", ".join(f"{name} {mean:+.3f}" for name, mean in zip(names, all_weights.mean(axis=0)))
        report(f"  mean weights        {weight_summary}")
        report(f"  mean cycles {np.mean(cycles_list):.0f}, reached TE target in "
               f"{reached} of {len(cycles_list)} runs, {seconds:.0f} seconds")

        for subject, subject_counts in zip(subjects, counts_by_subject):
            TP, FN, FP, TN = [int(c) for c in subject_counts]
            participant_file.write(f"{test},{set_name},{activation},{weight_text},{subject},{TP},{FN},{FP},{TN}\n")
        for row in weight_rows:
            weights_text = ",".join(f"{value:.6f}" for value in row[2:])
            weights_file.write(f"{test},{set_name},{activation},{weight_text},{row[0]},{row[1]},{weights_text}\n")
        participant_file.flush()
        weights_file.flush()

        summary.append((label, counts))

    # One line per experiment, rates computed from the added-up counts.
    report()
    report("Summary")
    report(f"{'experiment':<52}{'accuracy':>9}{'TPR':>8}{'TNR':>8}{'FPR':>8}{'FNR':>8}{'precision':>11}{'NoBuy acc':>11}")
    for label, counts in summary:
        TP, FN, FP, TN = [int(c) for c in counts]
        total = TP + FN + FP + TN
        report(f"{label:<52}{rate(TP + TN, total):>9}{rate(TP, TP + FN):>8}{rate(TN, FP + TN):>8}"
               f"{rate(FP, FP + TN):>8}{rate(FN, TP + FN):>8}{rate(TP, TP + FP):>11}{rate(FP + TN, total):>11}")

    participant_file.close()
    weights_file.close()
    results_file.close()
    print(f"\nResults saved to results_{tag}.txt, per_participant_{tag}.csv, weights_{tag}.csv")

# The 12 settings for one test: 3 input sets x hard/soft x weights on/off.
def all_settings(test):
    experiments = []
    for set_name in input_sets:
        for activation in ("hard", "soft"):
            for class_weights in (True, False):
                experiments.append((test, set_name, activation, class_weights))
    return experiments

# ---- Example learning and test patterns ----
# Saves one within-subject split (S01, fold 1, EEG + time) the way the
# neuron sees it: normalized inputs, then the label in the last column.
if run_within:
    columns = input_sets["EEG + time"]
    first = subjects[0]
    rows = features[features[:, 0] == first]
    folds = folds_main[first]
    train_rows = np.concatenate(folds[1:])
    test_rows = folds[0]
    low, spread = normalize_fit(rows[train_rows][:, columns])
    for name, picked in (("train", train_rows), ("test", test_rows)):
        patterns = np.column_stack((normalize_apply(rows[picked][:, columns], low, spread),
                                    rows[picked, 3]))
        np.savetxt(f"train_test_data/within_S{first:02d}_fold1_{name}.txt", patterns,
                   delimiter=",", fmt="%.6f", header="view_seconds,E,FAA,bought", comments="")

# ---- Stage 2 runs ----
if run_within:
    run_block("within", "Within-subject, stratified 5-fold, products viewed >= 1.0 s",
              features, folds_main, all_settings("within"))

if run_loso:
    run_block("loso", "Leave-one-subject-out, products viewed >= 1.0 s",
              features, folds_main, all_settings("loso"))

# Viewing time only, on every viewed product (quick glances included),
# to show how much of the decision is visible in looking time alone.
if run_extra:
    extra = []
    for test in ("within", "loso"):
        for activation in ("hard", "soft"):
            extra.append((test, "time only", activation, True))
    run_block("extra", "Viewing time only, all viewed products",
              features_all, folds_all, extra)

# =====================================================================
# Stage 3: tuning sweep
# =====================================================================

# Balanced accuracy: the average of the true positive rate and the true
# negative rate. 0.5 is guessing, 1.0 is perfect. Unlike plain accuracy,
# it cannot be raised by predicting NoBuy for everything.
def balanced_accuracy(counts):
    truePos, falseNeg, falsePos, trueNeg = counts
    return (truePos / (truePos + falseNeg) + trueNeg / (falsePos + trueNeg)) / 2

# Each setting is trained the same way as in Stage 2 (class weights on),
# then scored twice: on the training data and on the test data. The final
# values are chosen from the TRAINING scores only, so the test data plays
# no part in picking them. The test scores show how much the results move
# when a setting changes.
if run_sweep:
    start_results("results_sweep.txt")
    report_settings()
    report()
    report(f"Stage 3: tuning sweep, input set '{sweep_input_set}', class weights on")
    report("Final values are chosen from the training columns only.")

    sweep_file = open("sweep_results.csv", "w")
    sweep_file.write("test,activation,alpha,k,train_accuracy,train_TPR,train_TNR,train_balanced,"
                     "test_accuracy,test_TPR,test_TNR,test_precision,test_balanced\n")

    # Every setting to try: (activation, alpha, k). k is not used by hard.
    settings = [("hard", alpha, k) for alpha in sweep_alpha_hard]
    settings = settings + [("soft", alpha, gain) for alpha, gain in sweep_soft]
    columns = input_sets[sweep_input_set]

    for test in sweep_tests:
        report()
        report(f"{test}")
        report(f"{'':<22}{'---------- training ----------':>34}{'-------------- test --------------':>44}")
        report(f"{'activation':<11}{'alpha':>7}{'k':>4}"
               f"{'accuracy':>10}{'TPR':>8}{'TNR':>8}{'balanced':>10}"
               f"{'accuracy':>10}{'TPR':>8}{'TNR':>8}{'precision':>11}{'balanced':>10}{'seconds':>9}")

        for activation, alpha, gain in settings:
            start = time.time()
            counts_by_subject, weight_rows, cycles_list, reached, train_counts = run_experiment(
                features, folds_main, test, columns, activation, True, alpha, gain)
            seconds = time.time() - start

            test_counts = counts_by_subject.sum(axis=0)
            TP, FN, FP, TN = [int(c) for c in test_counts]
            trTP, trFN, trFP, trTN = [int(c) for c in train_counts]
            if activation == "hard":
                gain_text = "-"
            else:
                gain_text = f"{gain}"

            report(f"{activation:<11}{alpha:>7}{gain_text:>4}"
                   f"{rate(trTP + trTN, trTP + trFN + trFP + trTN):>10}{rate(trTP, trTP + trFN):>8}"
                   f"{rate(trTN, trFP + trTN):>8}{balanced_accuracy(train_counts):>10.4f}"
                   f"{rate(TP + TN, TP + FN + FP + TN):>10}{rate(TP, TP + FN):>8}{rate(TN, FP + TN):>8}"
                   f"{rate(TP, TP + FP):>11}{balanced_accuracy(test_counts):>10.4f}{seconds:>9.0f}")

            sweep_file.write(f"{test},{activation},{alpha},{gain_text},"
                             f"{rate(trTP + trTN, trTP + trFN + trFP + trTN).strip()},{rate(trTP, trTP + trFN).strip()},"
                             f"{rate(trTN, trFP + trTN).strip()},{balanced_accuracy(train_counts):.4f},"
                             f"{rate(TP + TN, TP + FN + FP + TN).strip()},{rate(TP, TP + FN).strip()},"
                             f"{rate(TN, FP + TN).strip()},{rate(TP, TP + FP).strip()},{balanced_accuracy(test_counts):.4f}\n")
            sweep_file.flush()

    sweep_file.close()
    results_file.close()
    print("\nResults saved to results_sweep.txt and sweep_results.csv")