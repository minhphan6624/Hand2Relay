import pandas as pd
import numpy as np

def check_csv_for_nan(csv_path):
    """
    Loads a CSV file, checks for NaN values, and prints basic statistics.
    """
    print(f"--- Checking {csv_path} for issues ---")
    try:
        df = pd.read_csv(csv_path)
        print(f"Shape of the dataset: {df.shape}")

        # Check for NaN values
        nan_count = df.isnull().sum().sum()
        if nan_count > 0:
            print(f"WARNING: Found {nan_count} NaN values in the dataset.")
            print("Columns with NaN values:")
            print(df.isnull().sum()[df.isnull().sum() > 0])
        else:
            print("No NaN values found in the dataset.")

        # Check for infinite values
        inf_count = np.isinf(df.select_dtypes(include=np.number)).sum().sum()
        if inf_count > 0:
            print(f"WARNING: Found {inf_count} infinite values in the dataset.")
        else:
            print("No infinite values found in the dataset.")

        # Display basic statistics
        print("\nBasic statistics (first 5 columns):")
        print(df.iloc[:, :5].describe())

        # Check for columns with zero standard deviation
        numeric_cols = df.select_dtypes(include=np.number).columns
        if 'label' in numeric_cols:
            numeric_cols = numeric_cols.drop('label')
        
        zero_std_cols = df[numeric_cols].std()
        zero_std_cols = zero_std_cols[zero_std_cols == 0]

        if not zero_std_cols.empty:
            print("\nWARNING: Found columns with zero standard deviation (all values are the same):")
            print(zero_std_cols)
            print("This can cause division-by-zero errors in StandardScaler.")
        else:
            print("\nNo columns with zero standard deviation found.")

        # Check for any rows that are entirely NaN (after dropping label)
        features_df = df.drop('label', axis=1)
        all_nan_rows = features_df.isnull().all(axis=1).sum()
        if all_nan_rows > 0:
            print(f"WARNING: Found {all_nan_rows} rows with all feature values as NaN.")


        # Print class distribution
        if 'label' in df.columns:
            print("\nClass distribution:")
            print(df['label'].value_counts().sort_index())
        else:
            print("\n'label' column not found for class distribution check.")

    except FileNotFoundError:
        print(f"Error: {csv_path} not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    check_csv_for_nan('src/data/landmarks_all.csv')
    check_csv_for_nan('src/data/landmarks_train.csv')
    check_csv_for_nan('src/data/landmarks_val.csv')
    check_csv_for_nan('src/data/landmarks_test.csv')
