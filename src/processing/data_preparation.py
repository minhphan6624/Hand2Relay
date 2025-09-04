import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def prepare_data_for_training(input_csv_path: str = '../src/data/landmarks_all.csv',
                              output_dir: str = '../src/data/'):
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

    # --- 2. Data Preprocessing: Normalization ---
    # Based on the PDF and model definition, the input size is 63 features (21 landmarks * 3 coords).
    # The existing data_preparation.py script added extra features, which would break the model.
    # Therefore, we will only apply normalization to the original 63 features.
    print("\n--- 2. Data Preprocessing: Normalization ---")
    X = df.drop('label', axis=1)
    y = df['label']

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
    # Using stratify to maintain class distribution
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_scaled_df, y, test_size=0.3, random_state=42, stratify=y)

    # Split temp into validation (15%) and test (15%)
    # test_size=0.5 means 50% of the temp set, which is 0.5 * 0.3 = 0.15 of the original set
    # Using stratify to maintain class distribution
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    print(f"Training set shape: {X_train.shape}, Labels shape: {y_train.shape}")
    print(f"Validation set shape: {X_val.shape}, Labels shape: {y_val.shape}")
    print(f"Test set shape: {X_test.shape}, Labels shape: {y_test.shape}")

    # Combine features and labels for saving
    train_df = pd.concat([X_train, y_train.reset_index(drop=True)], axis=1)
    val_df = pd.concat([X_val, y_val.reset_index(drop=True)], axis=1)
    test_df = pd.concat([X_test, y_test.reset_index(drop=True)], axis=1)

    # Save the datasets
    train_df.to_csv(os.path.join(output_dir, 'landmarks_train.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'landmarks_val.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'landmarks_test.csv'), index=False)

    print(f"Data splitting complete. Saved to:")
    print(f"- {os.path.join(output_dir, 'landmarks_train.csv')}")
    print(f"- {os.path.join(output_dir, 'landmarks_val.csv')}")
    print(f"- {os.path.join(output_dir, 'landmarks_test.csv')}")

if __name__ == "__main__":
    # This script is intended to be run as part of a larger process.
    # If you need to run it directly, uncomment the line below and ensure
    # 'src/data/landmarks_all.csv' exists.
    # prepare_data_for_training()
    pass
