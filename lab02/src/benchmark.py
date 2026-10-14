
import csv
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import sklearn

from main import prepare_data
from neural_network import initialize_parameters, forward, backward


def benchmark_network(width, X_batch, Y_batch, repetitions=100):
    # Use the required weight initialization.
    np.random.seed(42)
    parameters = initialize_parameters(
        input_size=64,
        hidden_size=width,
        output_size=10,
    )

    # Warm-up run before collecting timings.
    probabilities, cache = forward(X_batch, parameters)
    backward(Y_batch, parameters, cache)

    forward_times = []
    backward_times = []

    for _ in range(repetitions):
        # Measure forward propagation only.
        start = perf_counter()
        probabilities, cache = forward(X_batch, parameters)
        end = perf_counter()
        forward_times.append((end - start) * 1000)

        # Measure backward propagation only.
        start = perf_counter()
        gradients = backward(Y_batch, parameters, cache)
        end = perf_counter()
        backward_times.append((end - start) * 1000)

    return {
        "width": width,
        "batch_size": len(X_batch),
        "repetitions": repetitions,
        "forward_median_ms": float(np.median(forward_times)),
        "backward_median_ms": float(np.median(backward_times)),
    }


def main():
    print(f"Python: {sys.version.split()[0]}")
    print(f"NumPy: {np.__version__}")
    print(f"scikit-learn: {sklearn.__version__}")

    X_train, _, _, _, Y_train, _ = prepare_data()

    # Use one mini-batch of 32, matching Task 6.
    X_batch = X_train[:32]
    Y_batch = Y_train[:32]

    repetitions = 100
    results = []

    for width in [16, 64]:
        result = benchmark_network(
            width=width,
            X_batch=X_batch,
            Y_batch=Y_batch,
            repetitions=repetitions,
        )
        results.append(result)

        print(f"\nNetwork width: {width}")
        print(f"Batch size: {result['batch_size']}")
        print(f"Median forward time: "
              f"{result['forward_median_ms']:.6f} ms")
        print(f"Median backward time: "
              f"{result['backward_median_ms']:.6f} ms")

        if result["forward_median_ms"] > 0:
            ratio = (
                result["backward_median_ms"]
                / result["forward_median_ms"]
            )
            print(f"Backward / forward time ratio: {ratio:.2f}x")

    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    output_path = results_dir / "timing_results.csv"

    with output_path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "width",
            "batch_size",
            "repetitions",
            "forward_median_ms",
            "backward_median_ms",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nTiming results saved to: {output_path}")


if __name__ == "__main__":
    main()
