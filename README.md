# CryptOracle Benchmarking and Performance Analysis Repository  

## Overview  

This repository aims to be a standarizing benchmark framework for Homomorphic Encryption (HE). Current development focuses on implementing microbenchmarks for the OpenFHE library backend, implementing Logistical Regression and Resnet workload benchmarks for the OpenFHE library backend, saving the hardware/algorithm settings and associated results, and creating performance predictice models based on those results. This will inlcude both classical additive models and ML models for both the microbenchmarks and full workloads. Future development will be to add more software backends and support more hardware platforms (GPU).

### Clone the Repository  

```bash
git clone https://github.com/ParkerCMcLeod/unary-openfhe-benchmarking.git
cd unary-openfhe-benchmarking
```

## Features  

- **Benchmarking**: Analyze latency, power consumption, and throughput for HE microbenchmarks and workloads.  
- **Performance Profiling**: Generate FlameGraphs and conduct runtime analysis using `perf` tools.  
- **Configurable Parameters**: Fine-tune CKKS encryption parameters (security standards, `n`, depth, batch size, and OpenMP thread count) via CLI arguments.  

## System Dependencies  

### Required Packages  

- **Git**: For version control and repository management.  
- **CMake**: Build system tool for compiling the software.  
- **Autoconf**: Generates configuration scripts.  
- **Build-essential**: Installs necessary compilers and tools.  
- **Libtool**: Manages shared libraries.  
- **libgoogle-perftools-dev**: Provides the `tcmalloc` library for performance improvements.  
- **Linux-tools**: Kernel tools (e.g., `perf`) for performance monitoring.  
- **Python3-dev**: Development headers for building Python modules.  
- **Python3-pip**: Python package manager.  
- **Boost**: Collection of open-source, portable C++ libraries

### System Configuration  

Adjust `sysctl` settings to allow performance monitoring:  

```bash
# Temporarily set kernel.perf_event_paranoid to -1 for the current session
echo -1 | sudo tee /proc/sys/kernel/perf_event_paranoid
sudo sed -i '/^kernel\.perf_event_paranoid/d' /etc/sysctl.conf
echo "kernel.perf_event_paranoid = -1" | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
sudo sed -i '/^kernel\.kptr_restrict/d' /etc/sysctl.conf
echo "kernel.kptr_restrict = 0" | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

### Installation of System Dependencies 

#### Note: (WSL will not work - lack of full perf support 11/10/24)

```bash
sudo apt-get update -y && sudo apt-get upgrade -y
sudo apt install -y git cmake autoconf build-essential libtool \
  libgoogle-perftools-dev python3-dev python3-pip \
  linux-tools-$(uname -r) linux-tools-common linux-tools-generic
```

### Library Path Configuration

If you encounter "error while loading shared libraries" when running the benchmarks, you may need to set the `LD_LIBRARY_PATH` environment variable to include the OpenFHE library installation directory:

```bash
# Set LD_LIBRARY_PATH to include the OpenFHE libraries
export LD_LIBRARY_PATH=path/to/cloned/repository/openfhe-development-install/lib:$LD_LIBRARY_PATH
```

For persistence across terminal sessions, add this line to your `~/.bashrc` or `~/.profile` file:

```bash
echo 'export LD_LIBRARY_PATH="$HOME/path/to/unary-openfhe-benchmarking/openfhe-development-install/lib:$LD_LIBRARY_PATH"' >> ~/.bashrc
source ~/.bashrc
```

## Python Dependencies  

- **Python 3.x**  
- **Argparse**: Parse command-line arguments.  
- **Re**: Regular expression operations.  
- **Sys / OS / Subprocess**: System and OS interaction.  
- **Typing**: Type annotations (List, Tuple, Dict).  
- **Multiprocessing**: Parallel processing.  
- **Shutil**: High-level file operations.  
- **YAML / CSV / Datetime**: File parsing and time manipulation.  
- **Psutil**: System and process monitoring.  
- **Cpuinfo / GPUtil / Dmidecode**: Detailed hardware information (CPU, GPU, memory).  
- **PyTorch**: Deep learning framework.  
- **Pandas**: Data manipulation and analysis.  
- **Numpy**: Numerical computations.  
- **Matplotlib**: Visualization library.  

### Installation of Python Dependencies  

```bash
pip3 install torch pandas numpy matplotlib pyyaml
```

## Running Microbenchmarks  

```bash
python3 benchmark.py --power-latency-analysis True --runtime-analysis True --build True
```

## Complete CLI Example with Defaults  

```bash
python3 benchmark.py \
    -s none \
    -n 13 \
    -b 12 \
    --depth 5 \
    --num-threads 0 \
    --compiler-optimizations True \
    --build True \
    --run-primitives True \
    --csv-name test
    --verbose
```

## CLI Argument Reference  

| **Argument**                   | **Default Value** | **Description**                                                                               |
|--------------------------------|-------------------|-----------------------------------------------------------------------------------------------|
| `-s, --security-standard      `| `none`            | Sets CKKS security level (options: none, 128c, 192c, 256c, 128q, 192q, 256q).                 |
| `-n, --ring-dimension`         | `13`              | Sets the CKKS ring dimension (N) to 2^N (positive integer).                                   |
| `-b, --batch-size`             | `12`              | Sets the CKKS batch size to 2^b (positive integer, must be < N)                               |
| `-d, --depth`                  | `5`               | Sets the CKKS computation depth (positive integer).                                           |
| `-t, --num-threads`            | `0`               | Number of threads OpenFHE is permitted to use (0 uses all available threads).                 |
| `-p, --run-primitives`         | `False`           | Enables/disables primitive analysis (Primitives set in `in/primitives.yaml`).                 |
| `-p, --run-microbenchmarks`    | `False`           | Enables/disables microbenchmark analysis (Microbenchmarks set in `in/microbenchmarks.yaml`)   |
| `-p, --run-workloads`          | `False`           | Enables/disables workload analysis (Workloads set in `in/workloads.yaml`).                    |
| `-l, --power-latency-analysis` | `True`            | Enables/disables latency/power data collection. Must be `True` if runtime analysis is enabled.|
| `-r, --runtime-analysis`       | `True`            | Enables/disables runtime analysis.                                                            |
| `-f, --flamegraph-generation`  | `False`           | Enables/disables FlameGraph generation.                                                       |
| `-b, --build`                  | `True`            | Enables/disables build/rebuild (including checks).                                            |
| `-o, --compiler-optimizations` | `True`            | Enables/disables compiler optimizations.                                                      |
| `-e, --cold-caching`           | `True`            | Enables/disables cold-caching during primitive profiling.                                     |
| `-x, --csv-name`               | `""`              | Specifies "<level>-results-<csv-name>.csv" output file name.                                  |
| `-g, --run-group`              | `False`           | Should be `False` unless being called `util/run_group.py` or another script.                  |
| `-h, --fhe`                    | `False`           | Enabled only if bootstrapping primitive is being profiled.                                    |
| `--verbose`                    | `Disabled`        | Enables detailed logging.                                                                     |

## Further Documentation  

- **OpenFHE Documentation**: [OpenFHE Docs](https://openfhe-development.readthedocs.io/en/latest/index.html)

## Authors  
- Anonymous until conference submission results announced

## License  