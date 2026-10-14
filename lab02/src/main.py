
import random
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from neural_network import (initialize_parameters,forward,cross_entropy,)

# Reproducibility
random.seed(42)
np.random.seed(42)


def prepare_data():
    # Load the handwritten digits dataset
    X, y = load_digits(return_X_y=True)

    # Scale input features from 0–16 to 0–1
    X = X.astype(np.float64) / 16.0

    # One-hot encode the labels
    Y = np.eye(10)[y]

    # Stratified 70/30 split
    X_train, X_test, y_train, y_test, Y_train, Y_test = (
        train_test_split(
            X,
            y,
            Y,
            test_size=0.30,
            random_state=42,
            stratify=y,
        )
    )

    return X_train, X_test, y_train, y_test, Y_train, Y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, Y_train, Y_test = (
        prepare_data()
    )

    # Test both hidden-layer widths
    for width in [16, 64]:
        np.random.seed(42)

        parameters = initialize_parameters(hidden_size=width)

        # Use a small batch
        X_batch = X_train[:32]
        Y_batch = Y_train[:32]

        probabilities, cache = forward(X_batch, parameters)
        loss = cross_entropy(probabilities, Y_batch)

        print(f"\nHidden width: {width}")
        print("Prediction shape:", probabilities.shape)
        print("First probability row:", probabilities[0])
        print("Probability row sums:", probabilities.sum(axis=1)[:5])
        print("Initial loss:", loss)

        # Verify the required properties
        assert probabilities.shape == (32, 10)
        assert np.allclose(
            probabilities.sum(axis=1), 1.0, atol=1e-6
        )
        assert abs(loss - np.log(10)) < 0.01

        print("All forward-pass checks passed!")

