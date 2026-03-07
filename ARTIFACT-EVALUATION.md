# Appendix

## A. Abstract

This is a markdown form of the Artification Evaluation appendix found in the CryptOracle paper. This section describes how to access the artifacts of the CryptOracle framework introduced in Figure 3. It also describes how to reproduce all of the figures in the paper and collect new data representing Figures 4, 5, 7, 10 and 11.

## B. Artifact check-list (meta-information)

- Program: ISPASS 2026
- Compilation: GCC
- Run-time environment: C++, Python3
- Hardware: Linux-based CPUs, AMD and Intel
- Metrics: Latency, energy consumption, hardware events (via perf)
- Output: Profiling CSV reports, FlameGraph SVGs, and figure plots
- Experiments: Figures 4, 5, 7, 10 and 11
- How much disk space required (approximately)?: Negligible, < 1GB
- How much time is needed to prepare workflow (approximately)?: Expect ~30 minutes.
- How much time is needed to complete experiments (approximately)?: For figure generation, instantaneous. For reproducing profiling results, 3-4 hours.
- Publicly available?: Yes
- Code licenses (if publicly available)?: MIT License (LICENSE file in the GitHub repository)

## C. Description

### 1) How to access

A version of the CryptOracle repository that reproduces the paper result is available on Zennedo at: INSERT ZENEDO LINK. The most up-to-date version is also on GitHub at: https://github.com/UnaryLab/CryptOracle.

### 2) Hardware dependencies

- AMD or Intel CPU
- 16GB of RAM or more
- At least 1GB of disk space

### 3) Software dependencies

- Linux OS. Tested on Ubuntu 22.04. WSL will not work (missing full support for perf).
- Libraries: all required libraries listed in Installation section or handled by setup scripts.

## D. Installation

For artifact evaluation, please follow the below steps:

### 1. Clone the GitHub repository

```bash
git clone https://github.com/UnaryLab/CryptOracle/tree/ispass-ae
git checkout ispass-ae
cd CryptOracle
```

### 2. Install required dependencies

```bash
sudo apt-get update -y
sudo apt-get upgrade -y
sudo apt install -y git cmake autoconf build-essential libtool \
libgoogle-perftools-dev python3-dev python3-pip python3.12-venv \
libboost-all-dev linux-tools-\$(uname -r) linux-tools-common linux-tools-generic
```

### 3. Enable hardware performance counters

```bash
# Temporarily enable counters
echo -1 | sudo tee /proc/sys/kernel/perf_event_paranoid

# Persist across reboots
sudo sed -i '/^kernel\.perf_event_paranoid/d' /etc/sysctl.conf
echo "kernel.perf_event_paranoid = -1" | sudo tee -a /etc/sysctl.conf

# Allow unrestricted kernel symbol access
sudo sed -i '/^kernel\.kptr_restrict/d' /etc/sysctl.conf
echo "kernel.kptr_restrict = 0" | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

CryptOracle's profiler requires elevated kernel permissions to access hardware performance counters. If reviewers do not wish to grant these permissions, they may skip subsections F2 and F3.

### 4. Create a Python virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Clone and setup the primitive, microbenchmark, and workload directories

```bash
python3 benchmark-main.py --setup-benchmarking
```

## E. Experiment workflow

Now that the CryptOracle framework is set up, we will run the artifact to reproduce key results from the paper. It is not possible for us to provide direct access to the machines used to produce our results, but we can provide instructions to collect new profiling data on x86 CPUs, with the acknowledgement that there will be machine-specific variation in results.

## F. Evaluation and expected results

### 1) Recreating plots from collected data (all figures)

```bash
python3 ae-scripts/generate-ispass-plots.py
```

This will generate all of the figures from the paper from the included profiling datasets.

### 2) Collecting new primitive data (figure 5)

```bash
bash ae-scripts/collect_figure_5_data.sh
```

This script will collect profiling data for a sweep of security and thread parameters, specified under `ae-config/parameters.yaml`. This script follows the standard methodology for collecting results in the paper: collecting five runs for each parameter combination, then recording the median run.

### 3) Collecting new microbenchmark data (figure 7)

```bash
bash ae-scripts/collect_figure_7_data.sh
```

This example collects runtime, power, and perf event metrics for the matrix multiply operation, which is shown in Figure 7.

### 4) Using performance model (figures 10 and 11)

```bash
bash ae-scripts/estimate_matmul_performance.sh
```

This example involves using the performance model to estimate the performance of matrix multiplication based on the operation counts. This reproduces a subset of the results in Figures 10 and 11, corresponding to matrix multiplication. The figures and CSV will be produced in the `ae-plots` directory as `matrix_multiplication_analysis.png`, `matrix_multiplication_operation_contributions.png`, and `estimated_matrix_multiplication_median.csv`.

### 5) Generating FlameGraphs (figure 4)

```bash
bash ae-scripts/generate_figure_4_flamegraph.sh
```

To demonstrate CrypOracle's ability to automatically generate FlameGraphs for any primitive, microbenchmark, or workload, this example produces a FlameGraph for the Logistic Function microbenchmark, featured in Figure 4 of the paper. After running the command above, the flamegraph will be outputted to `out/flamegraphs` as `ae_logistic_function_FlameGraph.svg`.

## G. Experiment customization

To profile additional configurations, users can make modifications to the default yaml files in the `in/` directory. Additionally, they can create their own yaml config files and specify the path when calling `benchmark-main.py`. For the performance model, template config files are available under `perf-model` for estimating the performance of other supported microbenchmarks and workloads.
