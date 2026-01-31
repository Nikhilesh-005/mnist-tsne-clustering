
from sklearn.manifold import TSNE


def apply_tsne(X):
    """
    Reduces high-dimensional data into 2D using t-SNE.

    Args:
        X -> input features (n_samples, n_features)

    Returns:
        X_2d -> 2D representation of data
    """

    tsne = TSNE(
        n_components=2,     # We want 2D output
        random_state=42,    # Same result every run
        perplexity=30       # Controls how clusters form
    )

    X_2d = tsne.fit_transform(X)
    return X_2d

# Test this file independently
if __name__ == "__main__":
    import data_loader

    X, y, images = data_loader.load_mnist_data()
    X_2d = apply_tsne(X)

    print("t-SNE completed")
    print("2D shape:", X_2d.shape)
