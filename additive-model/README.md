CryptOracle Additive Model README

This module estimates application performance for a specified metric using an analytic performance model
based on primitive profiling results. It also optionally compares estimated performance against actual profiling.

Key Features:
    - Estimation of application performance using primitive operation benchmarks
    - Comparison of estimated vs actual performance across different parameters
    - Visualization of performance contributions by operation type
    - Support for different security standards and parameter configurations

Usage:
    python additive-model-estimation.py --config <yaml_config_path> --output-dir <output_path>

Example:
    python additive-model-estimation.py --config config.yaml --output-dir results/analysis

Configuration (YAML):
    Required fields:
        - name: Application name
        - primitive_benchmarks: Path to primitive benchmark results CSV
        - actual_benchmarks: Path to actual application benchmark results CSV
        - statistic: Statistical measure ('mean' or 'median')
        - title: Title for output files
        - operation_counts: Dictionary of operation names and their counts

Example YAML files for the three currently supported microbenchmarks and four workloads can be found under "microbenchmark-configs"
and "workload-configs", respectively. These files include:

/microbenchmark-configs/
    - matrix_multiplication_config.yaml
    - logistic_function_config.yaml
    - sign_eval_config.yaml
/workload-configs/
    - chi_square_test_config.yaml
    - cifar10_config.yaml
    - low_memory_resnet20_config.yaml
    - logistic_regression_config.yaml

Dependencies:
    - pandas: Data manipulation and analysis
    - matplotlib: Plotting and visualization
    - seaborn: Enhanced visualization
    - numpy: Numerical computations
    - scikit-learn: Statistical metrics
    - pyyaml: Configuration file parsing