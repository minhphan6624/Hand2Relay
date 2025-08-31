import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import os


def preprocess_and_visualize_data():
    """
    Loads, preprocesses, and visualizes hand gesture data.
    Generates gesture distribution, PCA, and t-SNE plots.
    """
    # Ensure results directory exists
    if not os.path.exists('../results'):
        os.makedirs('../results')

    # --- 1. Setup and Data Loading ---
    print("--- 1. Setup and Data Loading ---")
    try:
        df = pd.read_csv('../src/data/landmarks_all.csv')
        print("Dataset loaded successfully.")
        print(f"Shape of the dataset: {df.shape}")
        print("\nFirst 5 rows:")
        print(df.head())
    except FileNotFoundError:
        print("Error: landmarks_all.csv not found. Make sure the file is in the correct directory.")
        return
    except Exception as e:
        print(f"An error occurred during data loading: {e}")
        return

    # --- 2. Initial Data Exploration: Gesture Distribution ---
    print("\n--- 2. Initial Data Exploration: Gesture Distribution ---")
    gesture_counts = df['label'].value_counts()

    plt.figure(figsize=(10, 6))
    sns.barplot(x=gesture_counts.index,
                y=gesture_counts.values, palette='viridis')
    plt.title('Distribution of Hand Gestures')
    plt.xlabel('Gesture Label')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('../results/gesture_distribution.png')
    print("Gesture distribution plot saved to ../results/gesture_distribution.png")
    # plt.show() # In a script, we don't typically call plt.show() if saving

    # --- 3. Data Preprocessing: Feature Engineering and Normalization ---
    print("\n--- 3. Data Preprocessing: Feature Engineering and Normalization ---")
    X = df.drop('label', axis=1)
    y = df['label']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

    n_samples = X_scaled_df.shape[0]
    n_landmarks = 21
    n_features_per_landmark = 3

    X_reshaped = X_scaled_df.values.reshape(
        n_samples, n_landmarks, n_features_per_landmark)
    wrist_landmarks = X_reshaped[:, 0, :]

    distances_from_wrist = []
    for i in range(1, n_landmarks):
        landmark_i = X_reshaped[:, i, :]
        dist = np.linalg.norm(landmark_i - wrist_landmarks, axis=1)
        distances_from_wrist.append(dist)

    distances_df = pd.DataFrame(np.array(distances_from_wrist).T, columns=[
                                f'dist_wrist_lmk{i}' for i in range(1, n_landmarks)])
    X_processed = pd.concat([X_scaled_df, distances_df], axis=1)

    print("Data preprocessing complete.")
    print(f"Shape of processed data: {X_processed.shape}")
    print("\nFirst 5 rows of processed data:")
    print(X_processed.head())

    # --- 4. Dimensionality Reduction (PCA and t-SNE) ---
    print("\n--- 4. Dimensionality Reduction (PCA and t-SNE) ---")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_processed)
    pca_df = pd.DataFrame(data=X_pca, columns=['PCA1', 'PCA2'])
    pca_df['label'] = y.values

    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=300)
    X_tsne = tsne.fit_transform(X_processed)
    tsne_df = pd.DataFrame(data=X_tsne, columns=['tSNE1', 'tSNE2'])
    tsne_df['label'] = y.values

    print("Dimensionality reduction complete.")

    # --- 5. Visualization of Reduced Dimensions ---
    print("\n--- 5. Visualization of Reduced Dimensions ---")
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x='PCA1', y='PCA2', hue='label',
                    data=pca_df, palette='viridis', s=50, alpha=0.7)
    plt.title('PCA of Hand Gestures')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Gesture')
    plt.grid(True)
    plt.savefig('../results/pca_visualization.png')
    print("PCA visualization plot saved to ../results/pca_visualization.png")
    # plt.show()

    plt.figure(figsize=(10, 7))
    sns.scatterplot(x='tSNE1', y='tSNE2', hue='label',
                    data=tsne_df, palette='viridis', s=50, alpha=0.7)
    plt.title('t-SNE of Hand Gestures')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend(title='Gesture')
    plt.grid(True)
    plt.savefig('../results/tsne_visualization.png')
    print("t-SNE visualization plot saved to ../results/tsne_visualization.png")
    # plt.show()

    # --- 6. Conclusion ---
    print("\n--- 6. Conclusion ---")
    print("Data processing and visualization complete. Check the 'results/' directory for plots.")


if __name__ == "__main__":
    preprocess_and_visualize_data()
