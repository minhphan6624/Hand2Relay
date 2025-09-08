# Summary of NaN Problem and Fixes in Hand Gesture Recognition Project

This document details the troubleshooting process and resolution for a `NaN` (Not a Number) problem encountered during the training of the hand gesture recognition model.

## The Problem: `NaN` Loss During Training

Initially, attempts to train the neural network model resulted in immediate failure, characterized by:
*   **`Train Loss: nan` and `Val Loss: nan`**: The training and validation loss values became `NaN`, indicating a critical mathematical breakdown within the model's learning process. This meant the model was producing invalid numerical outputs and was unable to learn.
*   **Poor Classification Report**: The subsequent classification report showed near-zero precision, recall, and F1-scores for most gesture classes, confirming that the model had not learned to classify gestures effectively.

This "exploding loss" or `NaN` issue is a strong indicator of problems with the data being fed into the model.

## Troubleshooting Process and Solutions

We followed a systematic approach to identify and resolve the root cause:

### 1. Initial Hypothesis: Corrupted Raw Data

*   **Idea**: The simplest explanation was that the primary dataset, `src/data/landmarks_all.csv`, contained `NaN` values, infinite values, or other corrupted entries directly from the data collection phase.
*   **Investigation**: A diagnostic Python script (`check_data.py`) was created to load the CSV files, check for `NaN`s, infinite values, and display basic statistics.
*   **Finding**: `check_data.py` confirmed that `src/data/landmarks_all.csv` was clean, containing no `NaN`s or infinite values. However, the split datasets (`landmarks_train.csv`, `landmarks_val.csv`, `landmarks_test.csv`) *did* contain a significant number of `NaN` values, including entire rows where all feature values were `NaN`. This indicated the problem was introduced during the data preparation (splitting and scaling) phase.

### 2. Second Hypothesis: Zero Standard Deviation in Features

*   **Idea**: The `StandardScaler` (used for normalization) can produce `NaN`s if it encounters a feature column where all values are identical (i.e., the standard deviation is zero). Division by zero during scaling would result in `NaN`s.
*   **Investigation**: `check_data.py` was updated to specifically identify columns with zero standard deviation in the datasets.
*   **Finding**: The updated `check_data.py` reported "No columns with zero standard deviation found" in any of the datasets, including `landmarks_all.csv`. This ruled out the zero standard deviation issue as the cause.

### 3. Third Hypothesis: `train_test_split` Stratification Issues

*   **Idea**: The `train_test_split` function was being used with `stratify=y` to maintain class balance across the training, validation, and test sets. It was hypothesized that this stratification might be problematic, especially if some classes had very few samples, potentially leading to issues during splitting or the creation of empty/malformed splits.
*   **Investigation**: `src/processing/data_preparation.py` was modified to remove the `stratify=y` argument from both `train_test_split` calls, creating non-stratified splits.
*   **Finding**: Even after removing stratification, `check_data.py` still reported `NaN` values in the split datasets. This indicated that stratification was not the primary cause of the `NaN`s.

### 4. Final Hypothesis and The Solution: Index Misalignment During Concatenation

*   **Idea**: The most subtle and ultimately correct hypothesis was that `NaN`s were being introduced due to index misalignment when concatenating features (`X`) and labels (`y`) after the `train_test_split` operation. When `train_test_split` creates new dataframes, it preserves the original indices. If `X_train` and `y_train` (or `X_val`/`y_val`, `X_test`/`y_test`) had different, non-aligned indices after the split and shuffle, `pd.concat` would fill the missing entries with `NaN`s to align them.
*   **The Fix**: The `src/processing/data_preparation.py` script was modified to explicitly reset the index of both the feature dataframes (`X_train`, `X_val`, `X_test`) and the label series (`y_train`, `y_val`, `y_test`) *before* concatenating them. The `reset_index(drop=True)` method ensures that new, clean, sequential indices are assigned, allowing for proper row-wise concatenation.

    The problematic line:
    ```python
    train_df = pd.concat([X_train, y_train.reset_index(drop=True)], axis=1)
    ```
    Was changed to:
    ```python
    train_df = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
    ```
*   **Result**: After applying this fix and re-running `src/processing/data_preparation.py`, `check_data.py` confirmed that all split datasets were finally clean, with no `NaN` values. Subsequently, re-running `src/train/trainer.py` resulted in successful model training, with rapidly decreasing losses and near-perfect accuracy on both validation and test sets.

This iterative debugging process, focusing on data integrity at each step, successfully resolved the `NaN` problem and allowed for effective model training.
