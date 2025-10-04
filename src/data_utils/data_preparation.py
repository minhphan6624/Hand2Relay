import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
from pathlib import Path
from typing import List

from ..common.normalize_landmarks import normalize_landmarks

def augment_landmarks_single(landmarks_flat: np.ndarray, noise_std: float = 0.01, 
                             max_rotation_deg: float = 25.0) -> np.ndarray:
    """
    Applies random noise and rotation to a single set of normalized landmarks to create one augmented sample.
    Input:
        landmarks_flat: A 1D numpy array of 60 normalized landmark coordinates (x1,y1,z1, ...).
        noise_std: Standard deviation of Gaussian noise to add.
        max_rotation_deg: Maximum rotation angle in degrees.
    Output:
        A single augmented 1D numpy array.
    """
    original_landmarks_3d = landmarks_flat.reshape(-1, 3) # Reshape to (20, 3)

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

    return rotated_landmarks.flatten()

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

    # Ensure all feature columns are numeric before normalization
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.dropna() # Drop any rows that became NaN after conversion

    # Apply normalization to each row
    X_normalized_features = X.apply(normalize_landmarks, axis=1, result_type='expand')
    X_normalized_features = X_normalized_features.iloc[:, 3:] # Drop the first 3 columns (x0,y0,z0)

    # Create new names (x1,y1,z1,...,x20,y20,z20) for remaining columns
    new_feature_columns = [f"{ax}{i}" for i in range(1, 21) for ax in ("x", "y", "z")] 
    X_normalized_features.columns = new_feature_columns
    
    X = X_normalized_features

    print(f"Feature engineering completed. Shape of normalized features: {X.shape}")

    # --- 3. Data Augmentation (Targeted Oversampling) ---
    print("\n--- 3. Data Augmentation (Targeted Oversampling) ---")

    # Combine features and labels to easily work with them
    combined_df = pd.concat([X, y], axis=1)

    # Calculate class distribution
    class_counts = combined_df['label'].value_counts()
    max_samples = class_counts.max()
    print(f"Original class distribution:\n{class_counts}")
    print(f"Target sample count per class (balancing to max): {max_samples}")

    augmented_samples_list = [] # Renamed to avoid conflict with augment_landmarks_single output
    # Iterate over each class and its sample count
    for class_label, count in class_counts.items():
        # If the class is a minority class
        if count < max_samples:
            samples_to_generate = max_samples - count
            print(f"Class {class_label}: Augmenting with {samples_to_generate} new samples.")
            
            # Get all original samples for the current minority class
            minority_class_samples = combined_df[combined_df['label'] == class_label].drop('label', axis=1)
            
            # Generate the required number of new samples
            for _ in range(samples_to_generate):
                # Randomly pick one sample from the minority class to augment
                random_sample = minority_class_samples.sample(1).iloc[0].values
                
                # Generate ONE new augmented sample
                new_sample = augment_landmarks_single(random_sample) 
                
                # Store the new sample and its label
                augmented_samples_list.append(list(new_sample) + [class_label])

    # Create a new DataFrame from the augmented samples
    if augmented_samples_list:
        augmented_df = pd.DataFrame(augmented_samples_list, columns=combined_df.columns)
        
        # Concatenate the original data with the new augmented data
        balanced_df = pd.concat([combined_df, augmented_df], ignore_index=True)
    else:
        balanced_df = combined_df

    # Separate features and labels again
    X = balanced_df.drop('label', axis=1)
    y = balanced_df['label']

    print(f"Data augmentation complete. New balanced dataset shape: {X.shape}")
    print(f"New class distribution:\n{y.value_counts()}")

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

    # --- 6. Class Balance Inspection ---
    print("\n--- 6. Class Balance Inspection ---")
    print("Class balance in training set:")
    print(y_train.value_counts())
    print("\nClass balance in validation set:")
    print(y_val.value_counts())
    print("\nClass balance in test set:")
    print(y_test.value_counts())

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
