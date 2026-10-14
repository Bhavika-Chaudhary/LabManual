
import numpy as np


def initialize_parameters(input_size=64, hidden_size=16, output_size=10):
    """Initialize weights and biases for a three-layer network."""
    parameters = {
        "W1": np.random.normal(0, 0.01, (input_size, hidden_size)),
        "b1": np.zeros(hidden_size),
        "W2": np.random.normal(0, 0.01, (hidden_size, hidden_size)),
        "b2": np.zeros(hidden_size),
        "W3": np.random.normal(0, 0.01, (hidden_size, output_size)),
        "b3": np.zeros(output_size),
    }
    return parameters


def relu(z):
    """ReLU activation function."""
    return np.maximum(0, z)


def softmax(z):
    """Convert output scores into probabilities."""
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def forward(X, parameters):
    """Perform forward propagation through the network."""
    W1, b1 = parameters["W1"], parameters["b1"]
    W2, b2 = parameters["W2"], parameters["b2"]
    W3, b3 = parameters["W3"], parameters["b3"]

    # First hidden layer
    Z1 = X @ W1 + b1
    A1 = relu(Z1)

    # Second hidden layer
    Z2 = A1 @ W2 + b2
    A2 = relu(Z2)

    # Output layer
    Z3 = A2 @ W3 + b3
    probabilities = softmax(Z3)

    cache = {
        "X": X,
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2,
        "A2": A2,
        "Z3": Z3,
        "probabilities": probabilities,
    }

    return probabilities, cache


def cross_entropy(probabilities, Y):
    """Calculate mean cross-entropy loss."""
    epsilon = 1e-12
    probabilities = np.clip(probabilities, epsilon, 1.0)
    return -np.mean(np.sum(Y * np.log(probabilities), axis=1))


def backward(Y, parameters, cache):
    """Calculate gradients for all weights and biases."""
    X = cache["X"]
    Z1, A1 = cache["Z1"], cache["A1"]
    Z2, A2 = cache["Z2"], cache["A2"]
    probabilities = cache["probabilities"]

    W2 = parameters["W2"]
    W3 = parameters["W3"]

    m = X.shape[0]

    # Output layer: softmax + cross-entropy
    dZ3 = (probabilities - Y) / m
    dW3 = A2.T @ dZ3
    db3 = np.sum(dZ3, axis=0)

    # Second hidden layer
    dA2 = dZ3 @ W3.T
    dZ2 = dA2 * (Z2 > 0)
    dW2 = A1.T @ dZ2
    db2 = np.sum(dZ2, axis=0)

    # First hidden layer
    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * (Z1 > 0)
    dW1 = X.T @ dZ1
    db1 = np.sum(dZ1, axis=0)

    gradients = {
        "W1": dW1,
        "b1": db1,
        "W2": dW2,
        "b2": db2,
        "W3": dW3,
        "b3": db3,
    }

    return gradients
