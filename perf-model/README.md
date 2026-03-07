# CryptOracle Performance Model

## Overview

The **CryptOracle Performance Model** is a lightweight analytic tool for estimating and analyzing application performance using primitive-level benchmarks. It enables rapid design-space exploration and comparison of estimated vs. actual performance for FHE workloads.

## Key Features

* **Performance Estimation**: Predicts application runtime using primitive operation benchmarks
* **Comparison**: Visualizes estimated vs. actual performance across parameter sets
* **Operation Breakdown**: Shows performance contributions by operation type
* **Flexible Parameters**: Supports different security standards and configuration options

---

## Python Dependencies

| Package        | Purpose                        |
|---------------|---------------------------------|
| `pandas`      | Data manipulation and analysis  |
| `matplotlib`  | Plotting and visualization      |
| `seaborn`     | Enhanced visualization          |
| `numpy`       | Numerical computations          |
| `scikit-learn`| Statistical metrics             |
| `pyyaml`      | Configuration file parsing       |

---

## Usage

```bash
python3 perf-model.py --config <yaml_config_path> --output-dir <output_path>
```

**Example:**

```bash
cd perf-model/
python3 perf-model.py --config microbenchmark-configs/matrix_multiplication_config.yaml --output-dir results/
```

---

## Configuration (YAML)

**Required fields:**

| Field                | Description                                      |
|----------------------|--------------------------------------------------|
| `name`               | Application name                                 |
| `primitive_benchmarks` | Path to primitive benchmark results CSV        |
| `actual_benchmarks`  | Path to actual application benchmark results CSV |
| `statistic`          | Statistical measure (`mean` or `median`)         |
| `title`              | Title for output files                           |
| `operation_counts`   | Dictionary of operation names and their counts   |

---

## Example Config Files

Example YAML files for supported microbenchmarks and workloads are provided:

**Microbenchmarks:**

```
microbenchmark-configs/
    ├─ matrix_multiplication_config.yaml
    ├─ logistic_function_config.yaml
    └─ sign_eval_config.yaml
```

**Workloads:**

```
workload-configs/
    ├─ chi_square_test_config.yaml
    ├─ cifar10_config.yaml
    ├─ low_memory_resnet20_config.yaml
    └─ logistic_regression_config.yaml
```

---