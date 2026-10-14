# Lab 2: A Neural Network from Scratch

## 1. Goal

The goal of this lab was to implement a two-hidden-layer neural network from scratch using NumPy. The work included preparing the dataset, implementing forward propagation and cross-entropy loss, calculating gradients using backpropagation, validating gradients numerically, comparing training configurations, and measuring forward- and backward-pass execution times.

## 2. Method

### 2.1 Dataset Preparation

The `load_digits` dataset from scikit-learn was used. Each sample contains 64 input features representing an 8 × 8 image of a handwritten digit.

The input features were converted to `float64` and scaled by dividing by 16. The integer labels (0–9) were converted into one-hot encoded vectors containing 10 elements.

The dataset was split into 70% training data and 30% test data using a stratified split with `random_state=42`. Stratification maintains the class distribution across the training and test sets.

### 2.2 Neural Network Architecture

The network contains an input layer, two hidden layers, and an output layer:

**Input (64) → Dense → ReLU → Dense → ReLU → Dense (10) → Softmax**

Two hidden-layer widths were tested: W = 16 and W = 64.

The weight matrix shapes were:

- W1: (64, W)
- W2: (W, W)
- W3: (W, 10)

The corresponding bias shapes were (W,), (W,), and (10,).

Weights were initialized from a normal distribution N(0, 0.01), and biases were initialized to zero. ReLU activation was used in the hidden layers, while softmax converted the output logits into class probabilities.

### 2.3 Forward Propagation and Cross-Entropy Loss

Forward propagation was implemented using NumPy matrix multiplication. The input was passed through both hidden layers and then the output layer.

Cross-entropy loss was used to measure the difference between predicted probabilities and the correct one-hot encoded labels.

For a batch of 32 samples, the output shape was `(32, 10)`. The probability rows summed to 1 within the required tolerance of 10⁻⁶.

The initial losses were:

- W = 16: 2.302557
- W = 64: 2.302621

Both values were close to ln(10) ≈ 2.3026, as expected for an untrained network with approximately uniform output probabilities.

### 2.4 Backpropagation and Gradient Checking

Backpropagation was implemented using the chain rule to calculate gradients for all weight matrices and bias vectors.

Numerical gradient checking used central differences with a perturbation of 10⁻⁵. Ten selected weights were checked by comparing numerical gradients against the gradients calculated by backpropagation.

The final run passed 9 out of 10 checks, meeting the requirement that at least 9 of 10 relative errors be below 10⁻⁶. One check exceeded the threshold, with a relative error of approximately 5.970 × 10⁻⁶.

### 2.5 Training Experiments

The network was trained using mini-batches of 32 samples for 50 epochs. The training data were shuffled at the beginning of every epoch.

The following configurations were tested:

- Hidden widths: 16 and 64
- Learning rates: 0.001, 0.01, and 0.1

This resulted in six experiments. Training cross-entropy loss was recorded after each epoch. After training, the model's test accuracy was calculated using the held-out test set.

The loss curves were saved to `results/loss_curves.png`. Final training loss and test accuracy were saved to `results/accuracy.csv`, while additional loss and parameter diagnostics were saved to `results/diagnostics.csv`.

### 2.6 Execution Time Measurement

The forward and backward passes were timed separately using Python's `time.perf_counter()`.

A warm-up run was performed before collecting 100 timing measurements for each pass. The median execution time was reported for a batch size of 32 and hidden widths of 16 and 64.

The timing results were saved to `results/timing_results.csv`.

## 3. Results

### 3.1 Forward-Pass Validation

| Hidden Width | Output Shape | Initial Loss | Result |
|---:|---|---:|---|
| 16 | (32, 10) | 2.302557 | Passed |
| 64 | (32, 10) | 2.302621 | Passed |

Both configurations produced the required output shape, probability rows summing to 1, and initial losses close to ln(10).

### 3.2 Gradient Checking

| Measure | Result |
|---|---|
| Numerical perturbation | 10⁻⁵ |
| Relative-error threshold | 10⁻⁶ |
| Checks passed | 9/10 |
| Required minimum | 9/10 |
| Maximum relative error | 5.970 × 10⁻⁶ |
| Overall requirement | Passed |

The gradient check met the specified minimum. One of the ten checks exceeded the tolerance, while the other nine passed.

### 3.3 Training Results

| Hidden Width | Learning Rate | Final Training Loss | Test Accuracy |
|---:|---:|---:|---:|
| 16 | 0.001 | 2.302547 | 10.19% |
| 16 | 0.01 | 2.302475 | 10.19% |
| 16 | 0.1 | 1.744054 | 44.26% |
| 64 | 0.001 | 2.302517 | 10.19% |
| 64 | 0.01 | 2.302260 | 10.19% |
| 64 | 0.1 | 0.141502 | 93.33% |

The best-performing configuration was the network with hidden width 64 and learning rate 0.1, which achieved 93.33% test accuracy and a final training loss of 0.141502.

The learning rates 0.001 and 0.01 resulted in training losses that remained close to the initial loss, indicating very limited learning within 50 epochs under the specified initialization.

For width 16 with learning rate 0.1, the minimum recorded loss was 0.939228 at epoch 47, but the final loss increased to 1.744054. This indicates instability in the later training epochs. In contrast, the width-64 network with learning rate 0.1 reached its minimum recorded loss of 0.141502 at epoch 50.

The loss curves for all six experiments are shown in `results/loss_curves.png`.

### 3.4 Timing Results

| Hidden Width | Batch Size | Forward Median (ms) | Backward Median (ms) | Backward/Forward Ratio |
|---:|---:|---:|---:|---:|
| 16 | 32 | 0.040550 | 0.044300 | 1.09× |
| 64 | 32 | 0.103000 | 0.137000 | 1.33× |

The backward pass took longer than the forward pass for both network widths. For width 16, the backward-to-forward ratio was 1.09×. For width 64, it was 1.33×.

Both passes involve matrix multiplication. However, backpropagation also calculates gradients for the weights and biases and propagates errors through the network, requiring additional matrix operations. This helps explain why the backward pass took longer in these measurements.

Increasing the hidden width from 16 to 64 also increased the measured execution times for both passes.

## 4. Conclusions

1. **Forward propagation and loss calculation passed the required validation checks.** Both network widths produced outputs of the correct shape, probability rows summed to 1, and initial losses were close to ln(10). The gradient check also met the required minimum of 9 out of 10 passing checks.

2. **Training performance depended strongly on the hidden width and learning rate.** The width-64 network with learning rate 0.1 performed best, achieving 93.33% test accuracy. The smaller learning rates produced little improvement within 50 epochs, while the width-16 network with learning rate 0.1 showed a late increase in loss.

3. **Backward propagation took longer than forward propagation.** The backward-to-forward timing ratio increased from 1.09× for width 16 to 1.33× for width 64. The additional matrix multiplications needed to calculate and propagate gradients help explain the higher computational cost of backpropagation.

## 5. Reproducibility and Output Files

The reported environment used:

- Python 3.11.8
- NumPy 1.26.4
- scikit-learn 1.5.0

The data split used `random_state=42`, and random seeds were set in the scripts.

The generated result files are:

- `results/loss_curves.png` — training loss curves for all six configurations.
- `results/accuracy.csv` — final training loss and test accuracy.
- `results/diagnostics.csv` — minimum and maximum losses and parameter magnitudes.
- `results/timing_results.csv` — median forward- and backward-pass timings.