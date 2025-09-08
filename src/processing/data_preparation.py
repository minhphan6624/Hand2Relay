import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.common.models import normalize_landmarks

def augment_landmarks(landmarks_flat: np.ndarray, noise_std: float = 0.01, max_rotation_deg: float = 5.0, num_augmentations: int = 5) -> List[np.ndarray]:
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

def prepare_data_for_training(input_csv_path: str = 'src/data/landmarks_all.csv',
                              output_dir: str = 'src/data/'):
    """
    Loads, preprocesses (normalizes), and splits hand gesture data into training, validation, and test sets.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # --- 1. Setup and Data Loading ---
    print("--- 1. Setup and Data Loading ---")
    try:
        df = pd.read_csv(input_csv_path)
        print(f"Dataset loaded successfully from {input_csv_path}.")
        print(f"Shape of the dataset: {df.shape}")
        print("\nFirst 5 rows:")
        print(df.head())
    except FileNotFoundError:
        print(f"Error: {input_csv_path} not found. Make sure the file is in the correct directory.")
        return
    except Exception as e:
        print(f"An error occurred during data loading: {e}")
        return

    # --- 2. Data Preprocessing: Feature Engineering (Normalization) & Scaling ---
    print("\n--- 2. Data Preprocessing: Feature Engineering (Normalization) & Scaling ---")
    X = df.drop('label', axis=1)
    y = df['label']

    # Apply landmark normalization
    print("Applying landmark normalization (position and scale invariance)...")
    X_normalized_features = X.apply(normalize_landmarks, axis=1, result_type='expand')
    # The normalize_landmarks function returns 63 features, but we only need 60 (dropping x0,y0,z0)
    # So, we need to adjust the columns here.
    # Create new column names for the 60 features (x1,y1,z1 to x20,y20,z20)
    new_feature_columns = [f"{ax}{i}" for i in range(1, 21) for ax in ("x", "y", "z")]
    X_normalized_features = X_normalized_features.iloc[:, 3:] # Drop the first 3 columns (x0,y0,z0)
    X_normalized_features.columns = new_feature_columns
    X = X_normalized_features
    print("Landmark normalization complete and x0, y0, z0 columns dropped.")

    # --- Data Augmentation ---
    print("Applying data augmentation (noise and rotation)...")
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
    X = X.dropna() # Drop any rows that became NaN due to coercion

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

    print("Normalization complete.")
    print(f"Shape of normalized data: {X_scaled_df.shape}")
    print("\nFirst 5 rows of normalized data:")
    print(X_scaled_df.head())

    # --- 3. Data Splitting ---
    print("\n--- 3. Data Splitting ---")

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
    train_df.to_csv(os.path.join(output_dir, 'landmarks_train.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'landmarks_val.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'landmarks_test.csv'), index=False)

    print(f"Data splitting complete. Saved to:")
    print(f"- {os.path.join(output_dir, 'landmarks_train.csv')}")
    print(f"- {os.path.join(output_dir, 'landmarks_val.csv')}")
    print(f"- {os.path.join(output_dir, 'landmarks_test.csv')}")

if __name__ == "__main__":
    prepare_data_for_training()
