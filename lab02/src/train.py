
import csv
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import sklearn

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from neural_network import (
    initialize_parameters,
    forward,
    cross_entropy,
    backward,
)


def prepare_data():
    digits = load_digits()
    X = digits.data.astype(np.float64) / 16.0
    y = digits.target
    Y = np.eye(10)[y]

    X_train, X_test, y_train, y_test, Y_train, Y_test = train_test_split(
        X, y, Y,
        test_size=0.30,
        random_state=42,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test, Y_train, Y_test


def train_network(
    X_train,
    Y_train,
    width,
    learning_rate,
    epochs=50,
    batch_size=32,
    seed=42,
):
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    parameters = initialize_parameters(
        input_size=64,
        hidden_size=width,
        output_size=10,
    )

    loss_history = []
    parameter_max_history = []
    n_samples = X_train.shape[0]

    for epoch in range(epochs):
        indices = rng.permutation(n_samples)
        X_shuffled = X_train[indices]
        Y_shuffled = Y_train[indices]

        for start in range(0, n_samples, batch_size):
            end = start + batch_size
            X_batch = X_shuffled[start:end]
            Y_batch = Y_shuffled[start:end]

            probabilities, cache = forward(X_batch, parameters)
            gradients = backward(Y_batch, parameters, cache)

            for name in parameters:
                parameters[name] -= learning_rate * gradients[name]

        # Measure full training loss after the epoch's updates.
        train_probabilities, _ = forward(X_train, parameters)
        epoch_loss = cross_entropy(train_probabilities, Y_train)
        loss_history.append(float(epoch_loss))

        # Track the largest absolute parameter value.
        max_parameter = max(
            float(np.max(np.abs(value)))
            for value in parameters.values()
        )
        parameter_max_history.append(max_parameter)

        print(
            f"Width={width:2d}, LR={learning_rate:.3f}, "
            f"Epoch={epoch + 1:02d}/{epochs}, "
            f"Loss={epoch_loss:.6f}, "
            f"Max|parameter|={max_parameter:.6f}"
        )

        if not np.isfinite(epoch_loss) or not np.isfinite(max_parameter):
            raise FloatingPointError(
                f"Non-finite value detected at epoch {epoch + 1}, "
                f"width={width}, learning rate={learning_rate}"
            )

    assert len(loss_history) == epochs

    return parameters, loss_history, parameter_max_history


def main():
    random.seed(42)
    np.random.seed(42)

    print(f"Python: {sys.version.split()[0]}")
    print(f"NumPy: {np.__version__}")
    print(f"scikit-learn: {sklearn.__version__}")
    print(f"Matplotlib: {matplotlib.__version__}")

    X_train, X_test, y_train, y_test, Y_train, Y_test = prepare_data()

    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    widths = [16, 64]
    learning_rates = [0.001, 0.01, 0.1]
    epochs = 50
    batch_size = 32

    all_histories = {}
    accuracy_rows = []
    diagnostic_rows = []

    for width in widths:
        for learning_rate in learning_rates:
            print(
                f"\nTraining network: width={width}, "
                f"learning rate={learning_rate}"
            )

            parameters, losses, parameter_history = train_network(
                X_train=X_train,
                Y_train=Y_train,
                width=width,
                learning_rate=learning_rate,
                epochs=epochs,
                batch_size=batch_size,
                seed=42,
            )

            test_probabilities, _ = forward(X_test, parameters)
            predictions = np.argmax(test_probabilities, axis=1)
            accuracy = float(np.mean(predictions == y_test))

            label = f"Width={width}, LR={learning_rate}"
            all_histories[label] = losses

            accuracy_rows.append({
                "width": width,
                "learning_rate": learning_rate,
                "final_train_loss": losses[-1],
                "test_accuracy": accuracy,
            })

            min_epoch = int(np.argmin(losses)) + 1
            max_epoch = int(np.argmax(losses)) + 1

            diagnostic_rows.append({
                "width": width,
                "learning_rate": learning_rate,
                "minimum_loss": min(losses),
                "minimum_loss_epoch": min_epoch,
                "maximum_loss": max(losses),
                "maximum_loss_epoch": max_epoch,
                "final_loss": losses[-1],
                "maximum_absolute_parameter": max(parameter_history),
                "final_maximum_absolute_parameter": parameter_history[-1],
                "test_accuracy": accuracy,
            })

            print(f"Final training loss: {losses[-1]:.6f}")
            print(f"Minimum loss: {min(losses):.6f} at epoch {min_epoch}")
            print(f"Maximum loss: {max(losses):.6f} at epoch {max_epoch}")
            print(
                f"Maximum absolute parameter during training: "
                f"{max(parameter_history):.6f}"
            )
            print(f"Test accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    # Plot all six loss curves.
    fig, ax = plt.subplots(figsize=(11, 7))

    for label, losses in all_histories.items():
        ax.plot(
            range(1, epochs + 1),
            losses,
            label=label,
            linewidth=1.8,
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training cross-entropy loss")
    ax.set_title("Task 6: Training Loss for Different Network Configurations")
    ax.set_xlim(1, epochs)
    ax.set_xticks(range(0, epochs + 1, 5))
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=9)
    fig.tight_layout()

    plot_path = results_dir / "loss_curves.png"
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Save final training loss and test accuracy.
    csv_path = results_dir / "accuracy.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "width",
                "learning_rate",
                "final_train_loss",
                "test_accuracy",
            ],
        )
        writer.writeheader()
        writer.writerows(accuracy_rows)

    # Save loss and parameter diagnostics.
    diagnostics_path = results_dir / "diagnostics.csv"
    with diagnostics_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "width",
                "learning_rate",
                "minimum_loss",
                "minimum_loss_epoch",
                "maximum_loss",
                "maximum_loss_epoch",
                "final_loss",
                "maximum_absolute_parameter",
                "final_maximum_absolute_parameter",
                "test_accuracy",
            ],
        )
        writer.writeheader()
        writer.writerows(diagnostic_rows)

    print("\nTask 6 diagnostics completed.")
    print(f"Loss graph: {plot_path}")
    print(f"Accuracy table: {csv_path}")
    print(f"Diagnostics table: {diagnostics_path}")


if __name__ == "__main__":
    main()
