import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
from pathlib import Path
from typing import List

from ..common.normalize_landmarks import normalize_landmarks

def augment_landmarks(landmarks_flat: np.ndarray, noise_std: float = 0.01, 
                      max_rotation_deg: float = 25.0, num_augmentations: int = 5) -> List[np.ndarray]:
    """
    Applies random noise and rotation to a single set of normalized landmarks to create augmented samples.
    Input:
        landmarks_flat: A 1D numpy array of 60 normalized landmark coordinates (x1,y1,z1, ...).
        noise_std: Standard deviation of Gaussian noise to add.
        max_rotation_deg: Maximum rotation angle in degrees.
        num_augmentations: Number of augmented samples to generate per original sample.
    Output:
        A list of augmented 1D numpy arrays.
    """
    augmented_samples = []
    original_landmarks_3d = landmarks_flat.reshape(-1, 3) # Reshape to (20, 3)

    for _ in range(num_augmentations):
        # Add Gaussian noise
        noise = np.random.normal(0, noise_std, original_landmarks_3d.shape)
        noisy_landmarks = original_landmarks_3d + noise

        # Apply random rotation
        angle_x = np.random.uniform(-max_rotation_deg, max_rotation_deg)
        angle_y = np.random.uniform(-max_rotation_deg, max_rotation_deg)
        angle_z = np.random.uniform(-max_rotation_deg, max_rotation_deg)

        # Rotation matrices
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(np.deg2rad(angle_x)), -np.sin(np.deg2rad(angle_x))],
                       [0, np.sin(np.deg2rad(angle_x)), np.cos(np.deg2rad(angle_x))]])
        Ry = np.array([[np.cos(np.deg2rad(angle_y)), 0, np.sin(np.deg2rad(angle_y))],
                       [0, 1, 0],
                       [-np.sin(np.deg2rad(angle_y)), 0, np.cos(np.deg2rad(angle_y))]])
        Rz = np.array([[np.cos(np.deg2rad(angle_z)), -np.sin(np.deg2rad(angle_z)), 0],
                       [np.sin(np.deg2rad(angle_z)), np.cos(np.deg2rad(angle_z)), 0],
                       [0, 0, 1]])
        
        # Combine rotations (order matters: ZYX)
        R = Rz @ Ry @ Rx
        
        rotated_landmarks = np.dot(noisy_landmarks, R.T) # Apply rotation

        augmented_samples.append(rotated_landmarks.flatten())

    return augmented_samples

def run_pipeline(input_path: str = 'src/data/landmarks_all.csv',
                              output_dir: str = 'src/data/'):
    """
    Loads, preprocesses (normalizes), and splits hand gesture data into training, validation, and test sets.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # --- 1. Setup and Data Loading ---
    print("--- 1. Setup and Data Loading ---")
    
    df = pd.read_csv(input_path)
    print(f"Shape of the dataset: {df.shape}")
    print("First 5 rows of the dataset:")
    print(df.head())

    # --- 2. Feature Engineering (Normalization) ---
    print("\n--- 2. Data Preprocessing: Feature Engineering (Normalization) ---")

    X = df.drop('label', axis=1)
    y = df['label']

    # Apply normalization to each row
    X_normalized_features = X.apply(normalize_landmarks, axis=1, result_type='expand')
    X_normalized_features = X_normalized_features.iloc[:, 3:] # Drop the first 3 columns (x0,y0,z0)

    # Create new names (x1,y1,z1,...,x20,y20,z20) for remaining columns
    new_feature_columns = [f"{ax}{i}" for i in range(1, 21) for ax in ("x", "y", "z")] 
    X_normalized_features.columns = new_feature_columns
    
    X = X_normalized_features

    print(f"Feature engineering completed. Shape of normalized features: {X.shape}")

    # --- 3. Data Augmentation ---
    print("--- 3. Data Augmentation ---")
    augmented_X = []
    augmented_y = []
    
    # Iterate through each original sample and augment it
    for idx in range(len(X)):
        original_sample = X.iloc[idx].values
        original_label = y.iloc[idx]
        
        # Add original sample
        augmented_X.append(original_sample)
        augmented_y.append(original_label)

        # Generate augmented samples
        augmented_samples = augment_landmarks(original_sample, num_augmentations=5) # Generate 5 augmented samples
        for aug_sample in augmented_samples:
            augmented_X.append(aug_sample)
            augmented_y.append(original_label)

    X = pd.DataFrame(augmented_X, columns=X.columns)
    y = pd.Series(augmented_y, name='label')
    print(f"Data augmentation complete. New dataset shape: {X.shape}")

    # --- 4. Data Cleaning and Scaling ---
    print("\n--- 4. Data Cleaning and Scaling ---")

    # Drop rows with any NaN values that might have appeared during initial processing
    # This is crucial as StandardScaler cannot handle NaNs
    initial_rows = X.shape[0]
    
    df_cleaned = pd.concat([X, y], axis=1).dropna()
    X = df_cleaned.drop('label', axis=1)
    y = df_cleaned['label']
    if df_cleaned.shape[0] < initial_rows:
        print(f"Dropped {initial_rows - df_cleaned.shape[0]} rows containing NaN values.")

    # Ensure all feature columns are numeric before scaling
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.dropna() # Drop any rows that became NaN after conversion

    # Apply Standard Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

    print("Normalization complete.")
    print("\nFirst 5 rows of normalized data:")
    print(X_scaled_df.head())

    # --- 5. Data Splitting ---
    print("\n--- 5. Data Splitting ---")

    # Split into training (70%) and temp (30%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_scaled_df, y, test_size=0.3, random_state=42) 

    # Split temp into validation (15%) and test (15%)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42) 

    print(f"Training set shape: {X_train.shape}, Labels shape: {y_train.shape}")
    print(f"Validation set shape: {X_val.shape}, Labels shape: {y_val.shape}")
    print(f"Test set shape: {X_test.shape}, Labels shape: {y_test.shape}")

    # Combine features and labels for saving, ensuring index alignment
    train_df = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
    val_df = pd.concat([X_val.reset_index(drop=True), y_val.reset_index(drop=True)], axis=1)
    test_df = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)

    # Save the datasets
    train_output_path = Path(output_dir) / "landmarks_train.csv"
    val_output_path = Path(output_dir) / "landmarks_val.csv"
    test_output_path = Path(output_dir) / "landmarks_test.csv"

    train_df.to_csv(train_output_path, index=False)
    val_df.to_csv(val_output_path, index=False)
    test_df.to_csv(test_output_path, index=False)

    print(f"Data splitting complete. Saved to:")
    print(f"- Training set: {train_output_path}")
    print(f"- Validation set: {val_output_path}")
    print(f"- Test set: {test_output_path}")

if __name__ == "__main__":
    run_pipeline()
