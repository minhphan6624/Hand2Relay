# visualize_gesture_dataset.py
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

# === 1. Load and clean the dataset ===
path = "landmarks_all.csv"   # change if needed
df = pd.read_csv(path)

# Convert all numeric columns and drop invalid rows
df = df.apply(pd.to_numeric, errors='coerce').dropna()

# Separate features and labels
X = df.drop(columns=["label"]).values
y = df["label"].astype(int).values

print(f"✅ Loaded {len(df)} samples, {X.shape[1]} features per sample")
print("Class distribution:")
print(df["label"].value_counts().sort_index())

# === 2. Quick stats ===
print("\nFeature value ranges:")
print(df.describe().T[["mean", "std", "min", "max"]].head(10))

# === 3. PCA (for variance overview) ===
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(8,6))
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=y, palette="tab10", s=15)
plt.title("PCA projection (first 2 components)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(title="Gesture ID")
plt.tight_layout()
plt.show()

# === 4. t-SNE (for class separability) ===
print("\nRunning t-SNE (this can take 1–2 minutes)...")
tsne = TSNE(n_components=2, perplexity=40, learning_rate=200, n_iter=1000, random_state=42)
X_tsne = tsne.fit_transform(X)

plt.figure(figsize=(8,6))
sns.scatterplot(x=X_tsne[:,0], y=X_tsne[:,1], hue=y, palette="tab10", s=15)
plt.title("t-SNE visualization of gesture clusters")
plt.xlabel("t-SNE Dim 1")
plt.ylabel("t-SNE Dim 2")
plt.legend(title="Gesture ID")
plt.tight_layout()
plt.show()

# === 5. Optional: Check overlap ===
# Compute average pairwise distance between cluster centroids
import numpy.linalg as LA
centroids = np.array([X[y==i].mean(axis=0) for i in np.unique(y)])
distances = LA.norm(centroids[:,None,:] - centroids[None,:,:], axis=2)
print("\nAverage centroid distances between gestures:")
print(pd.DataFrame(distances, 
                   index=np.unique(y), 
                   columns=np.unique(y)).round(3))
