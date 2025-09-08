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

### 5. Challenge: Overfitting and Lack of Generalization (Initial Simulation Failure)

*   **Problem**: After resolving the `NaN` issue and achieving 100% accuracy on the test set, the model performed poorly in live webcam simulation. It struggled to recognize gestures when the hand's position or size varied, often defaulting to one or two classes.
*   **Analysis**: This was a classic case of **overfitting**. The model had memorized the exact landmark coordinates from the training data, rather than learning the general *shape* of the gestures. The test set was likely too similar to the training set, giving a false sense of high performance.
*   **Initial Proposed Solution**: Implement **Feature Engineering** by normalizing landmarks relative to the wrist (landmark 0) and scaling them to make them invariant to hand position and size.

### 6. Solution: Feature Engineering (Landmark Normalization)

*   **Idea**: To make the model robust to variations in hand position and size, we needed to transform the raw 63 `(x,y,z)` landmark coordinates into a position- and scale-invariant representation.
*   **Implementation**: A `normalize_landmarks` function was introduced. This function:
    *   Translated all landmarks so the wrist (landmark 0) became the origin `(0,0,0)`.
    *   Scaled all landmarks based on the distance from the wrist to the middle finger MCP joint (landmark 9), making the gesture size-invariant.
*   **Location**: This function was initially added to `src/processing/data_preparation.py`.

### 7. Challenge: Mismatched Preprocessing (Training vs. Inference)

*   **Problem**: A critical realization was that the `normalize_landmarks` function was only being applied during data preparation (for training). If the same normalization was not applied to the live webcam data during inference (`main_controller.py`), the model would receive raw, unnormalized data, leading to a mismatch and poor performance.
*   **Analysis**: The preprocessing pipeline for training and inference must be identical for a machine learning model to generalize correctly.
*   **Solution**: Centralize the `normalize_landmarks` function and apply it consistently.

### 8. Solution: Centralizing Normalization and Applying to Inference

*   **Implementation**:
    *   The `normalize_landmarks` function was moved from `src/processing/data_preparation.py` to `src/common/models.py`, making it a shared utility.
    *   `src/processing/data_preparation.py` was updated to import `normalize_landmarks` from `src/common/models.py`.
    *   `main_controller.py` was updated to import `normalize_landmarks` from `src/common/models.py` and apply it to the raw webcam landmarks *before* feeding them to the model.

### 9. Challenge: Zero Standard Deviation in Normalized Wrist Features

*   **Problem**: After applying `normalize_landmarks`, the `x0, y0, z0` coordinates (representing the wrist) became `(0,0,0)` for every sample. When `StandardScaler` was applied, it encountered columns with zero standard deviation, which can cause numerical instability or `NaN`s.
*   **Analysis**: These `x0, y0, z0` features became redundant after normalization, as they no longer carried unique information.
*   **Solution**: Drop the redundant features.

### 10. Solution: Dropping Redundant Wrist Features

*   **Implementation**: `src/processing/data_preparation.py` was modified to explicitly drop the `x0, y0, z0` columns from the feature set `X` immediately after applying `normalize_landmarks` and before scaling. This reduced the feature dimensionality from 63 to 60.

### 11. Challenge: Model Input Size Mismatch

*   **Problem**: After reducing the feature set to 60, the model training failed with a `RuntimeError` (`mat1 and mat2 shapes cannot be multiplied`). The model's first linear layer was still expecting 63 input features, while the data provided had 60.
*   **Analysis**: Although `src/common/models.py`'s `HandGestureClassifier` had its default `input_size` changed to 60, `src/train/trainer.py` was explicitly overriding this with `input_size=63` during model initialization.
*   **Solution**: Update Model Input Size in Trainer.

### 12. Solution: Updating Model Input Size in Trainer

*   **Implementation**: `src/train/trainer.py` was modified to change the `input_size` from 63 to 60 when initializing the `HandGestureClassifier`.

### 13. Current Challenge: Orientation Sensitivity (Still Overfitting)

*   **Problem**: Even after all previous fixes, the live simulation shows that the model still requires the hand to be in a specific orientation (e.g., fingers pointing up) to correctly classify gestures. It struggles with rotated or tilted hands.
*   **Analysis**: This indicates that while position and scale invariance are achieved, the model is still overfitting to the specific orientations present in the original (and gently augmented) training data. The current data augmentation (5 degrees rotation) is too conservative.
*   **Proposed Solution**: Implement more aggressive data augmentation.

### 14. Proposed Solution: Aggressive Data Augmentation

*   **Idea**: To make the model robust to variations in hand orientation, we need to significantly increase the diversity of rotated examples in the training data.
*   **Plan**: Modify the `augment_landmarks` function in `src/processing/data_preparation.py` to increase `max_rotation_deg` from `5.0` to `25.0` degrees. This will generate augmented samples with a much wider range of rotations, forcing the model to learn orientation-invariant features.
*   **Next Steps**: Re-run data preparation, re-train the model, and re-test in simulation.
