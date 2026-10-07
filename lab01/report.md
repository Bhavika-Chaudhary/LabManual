# Lab 1 — Environment and First System Measurements

## 1. Goal

The goal of this laboratory work is to create a reproducible Python environment, train two baseline machine learning models, measure their system costs, and evaluate their suitability for different deployment environments.

The two baseline models used in this laboratory are Logistic Regression and Random Forest. The measured system costs include training time, single-sample inference latency, model size, and peak memory usage.

The final objective is to compare the measured models with Cloud, Edge, Mobile, and TinyML deployment requirements.

## 2. Method

### 2.1 Environment

The experiment was performed using Python 3.11.8 in an isolated virtual environment named `.venv`.

The required Python packages were installed from `requirements.txt` with pinned versions to improve reproducibility.

Python and package versions were recorded using `src/print_versions.py` and saved to:

`results/versions.txt`

The experiment used fixed random seeds for reproducibility:

- `random.seed(42)`
- `numpy.random.seed(42)`
- `torch.manual_seed(42)`
- `torch.cuda.manual_seed_all(42)`

### 2.2 Dataset

The Breast Cancer Wisconsin (Diagnostic) dataset was used.

The dataset was loaded using `load_breast_cancer(return_X_y=True)`.

The dataset was divided into training and testing sets using a 70/30 stratified split with `random_state=42`.

This ensures that the class distribution is preserved between the training and testing sets and that the experiment can be reproduced.

### 2.3 Baseline Models

Two classification models were trained:

1. Logistic Regression
2. Random Forest

Logistic Regression was configured with:

- `max_iter=1000`
- `random_state=42`

Random Forest was configured with:

- `n_estimators=100`
- `random_state=42`

Test accuracy was calculated using the held-out test dataset and recorded to four decimal places.

### 2.4 System Measurements

The following system measurements were collected for both models:

- training time
- single-sample inference latency
- model size
- peak memory usage

#### Training time

One warm-up training run was performed first.

After the warm-up, five training runs were measured. The median of the five measured runs was reported.

#### Inference latency

A single test sample was selected from the test set.

The model performed 100 single-sample inference operations. The median inference time was reported in milliseconds.

#### Model size

Each trained model was saved using `joblib`.

The resulting file size was measured in bytes and converted to KB.

#### Peak memory

Peak memory usage was measured during model training and inference.

## 3. Results

### 3.1 Baseline Accuracy

| Model | Test Accuracy |
|---|---:|
| Logistic Regression | 0.9415 (94.15%) |
| Random Forest | 0.9357 (93.57%) |

Logistic Regression achieved the higher test accuracy in this experiment, with 94.15% compared with 93.57% for Random Forest.

The difference in accuracy is small, with Logistic Regression performing approximately 0.58 percentage points better.

### 3.2 System Measurements

| Model | Training Time (s) | Inference Latency (ms) | Model Size (KB) | Peak Inference Memory (MiB) |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.3993 | 0.3684 | 1.03 | 228.70 |
| Random Forest | 0.4603 | 9.7427 | 284.07 | 229.49 |

Logistic Regression had the lower training time and substantially lower inference latency.

Random Forest required more time for training and inference and produced a much larger model file.

The two models had similar peak inference memory usage.

### 3.3 Deployment Suitability

The measured models were compared with the deployment budgets specified in the laboratory manual.

| Model | Cloud | Edge | Mobile | TinyML |
|---|---|---|---|---|
| Logistic Regression | PASS | PASS | PASS | FAIL |
| Random Forest | PASS | PASS | PASS | FAIL |

#### Cloud

Both models satisfy the specified Cloud deployment requirements.

Their measured inference latencies are below the 100 ms limit and their model sizes are below the 500 MB limit.

#### Edge

Both models satisfy the specified Edge deployment requirements.

Their inference latencies are below the 50 ms limit and their model sizes are below the 50 MB limit.

#### Mobile

Both models satisfy the specified Mobile deployment requirements.

Logistic Regression has an inference latency of 0.3684 ms and a model size of 1.03 KB.

Random Forest has an inference latency of 9.7427 ms and a model size of 284.07 KB.

Both models are below the specified Mobile latency and model-size limits.

Their measured peak inference memory is also within the specified Mobile memory range.

#### TinyML

Neither model satisfies the TinyML requirements.

Logistic Regression satisfies the TinyML latency and model-size limits, but its measured peak inference memory of approximately 228.70 MiB is far above the TinyML memory limit.

Random Forest also satisfies the TinyML latency limit, but its model size of 284.07 KB exceeds the specified model-size limit. Its measured peak inference memory of approximately 229.49 MiB is also far above the TinyML memory limit.

Therefore, memory is the main limitation for both models, while Random Forest also exceeds the TinyML model-size limit.

## 4. Discussion

The results demonstrate that model accuracy alone is not sufficient when selecting a model for deployment.

Logistic Regression achieved slightly higher test accuracy than Random Forest while also having a substantially lower inference latency and much smaller model size.

The median inference latency of Logistic Regression was 0.3684 ms, compared with 9.7427 ms for Random Forest.

The model-size difference was also significant. Logistic Regression required only 1.03 KB, whereas Random Forest required 284.07 KB.

The two models had similar measured peak inference memory usage, at approximately 229 MiB.

This shows that a smaller model file does not necessarily mean proportionally lower runtime memory usage.

Both models are suitable for the less restrictive Cloud and Edge environments and satisfy the specified Mobile requirements. However, neither model is suitable for TinyML in the measured configuration because of the memory requirement.

## 5. Conclusions

1. **Logistic Regression was the more efficient model in this experiment.** It achieved slightly higher test accuracy than Random Forest while having lower training time, much lower inference latency, and a substantially smaller model size.

2. **Both models are suitable for Cloud, Edge, and Mobile deployment according to the specified deployment budgets.** Their measured latency, model size, and memory requirements satisfy the corresponding limits.

3. **Neither model is suitable for TinyML in the current configuration.** The main limitation is peak memory usage. Random Forest also exceeds the TinyML model-size limit.

## 6. Reproducibility

The experiment uses Python 3.11.8, pinned package versions, fixed random seeds, and a fixed 70/30 stratified train/test split with `random_state=42`.

The generated experimental evidence is stored in the `results` directory, including:

- `versions.txt`
- `baseline_accuracy.csv`
- `system_costs.csv`
- saved model files

The source scripts used to perform the experiments are stored in the `src` directory.