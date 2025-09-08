import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

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

    # --- 2. Data Preprocessing: Normalization ---
    print("\n--- 2. Data Preprocessing: Normalization ---")
    X = df.drop('label', axis=1)
    y = df['label']

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
