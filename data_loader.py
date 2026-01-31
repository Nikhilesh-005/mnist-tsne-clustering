
from sklearn.datasets import load_digits
import numpy as np


def load_mnist_data():

    # Load dataset
    digits = load_digits()

    # Each image converted into numbers
    X = digits.data         
    y = digits.target        
    images = digits.images   

    # Normalize pixel values (0 to 1 range)
    X = X / 16.0

    return X, y, images


if __name__ == "__main__":
    X, y, images = load_mnist_data()
    print("MNIST data loaded")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("images shape:", images.shape)
