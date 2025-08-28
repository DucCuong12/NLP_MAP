from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
  

def draw_pca(embeddings, labels):
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)

    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=encoded_labels, cmap="nipy_spectral", alpha=0.8, s=10)
    plt.colorbar(scatter, label="Labels")
    plt.title("PCA Embedding Visualization with Labels")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)
    plt.show()