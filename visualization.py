
import plotly.express as px
import pandas as pd


def plot_clusters(X_2d, true_labels, cluster_labels):
    """
    Creates an interactive 2D scatter plot using Plotly.

    Args:
        X_2d           -> 2D t-SNE output
        true_labels    -> Actual digit labels (0–9)
        cluster_labels -> K-Means cluster assignments
    """

    df = pd.DataFrame({
        "x": X_2d[:, 0],
        "y": X_2d[:, 1],
        "Digit": true_labels,
        "Cluster": cluster_labels
    })

    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="Cluster",
        hover_data=["Digit", "Cluster"],
        title="MNIST Digit Clustering using t-SNE + K-Means"
    )

    return fig



# Test visualization independently
if __name__ == "__main__":
    import data_loader
    import tsne_utils
    import clustering

    X, y, images = data_loader.load_mnist_data()
    X_2d = tsne_utils.apply_tsne(X)
    cluster_labels, score = clustering.apply_kmeans(X_2d, k=10)

    plot_clusters(X_2d, y, cluster_labels)
