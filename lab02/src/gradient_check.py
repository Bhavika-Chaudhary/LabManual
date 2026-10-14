
import sys
import numpy as np
import sklearn

from main import prepare_data
from neural_network import initialize_parameters, forward, backward


EPSILON = 1e-5
TOLERANCE = 1e-6
NUM_CHECKS = 10
SEED = 42


def loss_from_logits(logits, Y):
    """Calculate cross-entropy directly from logits with high precision."""
    logits = np.asarray(logits, dtype=np.longdouble)
    Y = np.asarray(Y, dtype=np.longdouble)

    # Stable log-sum-exp calculation.
    row_max = np.max(logits, axis=1, keepdims=True)
    shifted = logits - row_max

    log_sum_exp = (
        np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
        + row_max
    )

    log_probabilities = logits - log_sum_exp
    return float(-np.mean(np.sum(Y * log_probabilities, axis=1)))


def numerical_gradient(X, Y, parameters, name, index, epsilon):
    """Estimate one parameter's gradient using central differences."""
    tensor = parameters[name]
    original = tensor[index].copy()

    try:
        tensor[index] = original + epsilon
        _, cache_plus = forward(X, parameters)
        loss_plus = loss_from_logits(cache_plus["Z3"], Y)

        tensor[index] = original - epsilon
        _, cache_minus = forward(X, parameters)
        loss_minus = loss_from_logits(cache_minus["Z3"], Y)

    finally:
        tensor[index] = original

    return (loss_plus - loss_minus) / (2.0 * epsilon)


def gradient_check():
    np.random.seed(SEED)

    X_train, _, _, _, Y_train, _ = prepare_data()

    # A small, fixed batch makes the check reproducible.
    X_batch = X_train[:20]
    Y_batch = Y_train[:20]

    parameters = initialize_parameters(
        input_size=64,
        hidden_size=16,
        output_size=10,
    )

    probabilities, cache = forward(X_batch, parameters)
    analytical_gradients = backward(Y_batch, parameters, cache)

    # Check gradient dimensions and numerical validity first.
    for name, parameter in parameters.items():
        gradient = analytical_gradients[name]

        assert gradient.shape == parameter.shape, (
            f"Incorrect gradient shape for {name}"
        )
        assert np.all(np.isfinite(gradient)), (
            f"Non-finite gradient found for {name}"
        )

    # The lab asks us to check weights, so sample from W1, W2 and W3.
    rng = np.random.default_rng(SEED)
    weight_names = ["W1", "W2", "W3"]

    candidates = []
    for name in weight_names:
        for index in np.ndindex(parameters[name].shape):
            candidates.append((name, index))

    selected = rng.choice(len(candidates), size=NUM_CHECKS, replace=False)

    errors = []

    print("TASK 5: NUMERICAL GRADIENT CHECK")
    print(f"Python version: {sys.version.split()[0]}")
    print(f"NumPy version: {np.__version__}")
    print(f"scikit-learn version: {sklearn.__version__}")
    print(f"Epsilon: {EPSILON:g}")
    print(f"Relative-error tolerance: {TOLERANCE:g}")
    print("-" * 65)

    for selected_index in selected:
        name, index = candidates[int(selected_index)]

        numerical = numerical_gradient(
            X_batch, Y_batch, parameters, name, index, EPSILON
        )
        analytical = float(analytical_gradients[name][index])

        denominator = max(
            1e-12,
            abs(numerical) + abs(analytical),
        )
        relative_error = abs(numerical - analytical) / denominator
        errors.append(relative_error)

        status = "PASS" if relative_error < TOLERANCE else "FAIL"
        print(
            f"{name}{index}: "
            f"numerical={numerical:.10e}, "
            f"analytical={analytical:.10e}, "
            f"relative error={relative_error:.3e} [{status}]"
        )

    passed = sum(error < TOLERANCE for error in errors)

    print("-" * 65)
    print(f"Gradient checks passed: {passed}/{NUM_CHECKS}")
    print(f"Maximum relative error: {max(errors):.3e}")

    if passed >= 9:
        print("POINT 5 GRADIENT CHECK: PASSED")
    else:
        print("POINT 5 GRADIENT CHECK: NOT YET PASSED")
        print("Inspect the failed coordinates and gradient calculations.")

    return passed


if __name__ == "__main__":
    gradient_check()
