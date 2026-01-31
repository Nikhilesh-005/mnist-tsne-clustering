
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def apply_kmeans(X_2d, k):

    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_2d)

    score = silhouette_score(X_2d, labels)

    return labels, score


# Test this file independently
if __name__ == "__main__":
    import data_loader
    import tsne_utils

    X, y, images = data_loader.load_mnist_data()
    X_2d = tsne_utils.apply_tsne(X)

    labels, score = apply_kmeans(X_2d, k=10)

    print("Clustering completed")
    print("Number of clusters:", len(set(labels)))
    print("Silhouette score:", round(score, 3))

    
def get_misclassified_indices(true_labels, cluster_labels):
    """
    Finds indices of misclassified samples based on majority label per cluster.
    """

    misclassified = []

    clusters = np.unique(cluster_labels)

    for cluster in clusters:
        indices = np.where(cluster_labels == cluster)[0]
        cluster_true_labels = true_labels[indices]

        # Majority digit in this cluster
        majority_label = np.bincount(cluster_true_labels).argmax()

        for idx in indices:
            if true_labels[idx] != majority_label:
                misclassified.append(idx)

    return misclassified
