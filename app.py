
import streamlit as st
import data_loader
import tsne_utils
import clustering
import visualization


st.title("MNIST Digit Clustering with t-SNE + K-Means")

st.write("""
This app visualizes handwritten digit clusters using:
- t-SNE for dimensionality reduction
- K-Means for clustering
""")


# Slider for number of clusters
k = st.slider(
    "Select number of clusters (k)",
    min_value=3,
    max_value=15,
    value=10
)


# Load data
X, y, images = data_loader.load_mnist_data()

st.write("MNIST dataset loaded")


# Apply t-SNE
with st.spinner("Running t-SNE..."):
    X_2d = tsne_utils.apply_tsne(X)


# Apply clustering
cluster_labels, silhouette = clustering.apply_kmeans(X_2d, k)


# Display metrics
st.subheader("Clustering Metrics")
st.write("Silhouette Score:", round(silhouette, 3))


# Plot visualization
st.subheader("Cluster Visualization")
fig = visualization.plot_clusters(X_2d, y, cluster_labels)
st.plotly_chart(fig)

st.subheader("Misclassified Digits")

show_misclassified = st.checkbox("Show misclassified digits")

if show_misclassified:
    misclassified_indices = clustering.get_misclassified_indices(y, cluster_labels)

    st.write(f"Total misclassified digits: {len(misclassified_indices)}")

    # Show first 25 misclassified digits
    cols = st.columns(5)
    for i, idx in enumerate(misclassified_indices[:25]):
        with cols[i % 5]:
            st.image(images[idx] / images[idx].max(), caption=f"True: {y[idx]}", width=80)

