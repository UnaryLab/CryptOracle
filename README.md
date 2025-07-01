# CryptOracle Benchmarking and Performance Analysis Repository  

### Overview  

**CryptoOracle** is an **open-source, fully modular framework** for the **systematic characterization, profiling, and performance modeling** of workloads that rely on **Fully Homomorphic Encryption (FHE)**.  
Built on top of the community-maintained **[OpenFHE](https://openfhe-development.readthedocs.io/en/latest/index.html)** library, with a focus on the CKKS approximate-arithmetic scheme, CryptoOracle streamlines **reproducible evaluation** for cryptography researchers, system architects, and ML practitioners.

CryptoOracle is composed of three tightly-integrated pillars:

| Component             | Purpose                                                                                                                                                                                                                                                                                            |
| --------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Benchmark Suite**   | A top-down collection spanning<br>• **Workloads** - e.g., Logistic Regression, ResNet-20 image classification<br>• **Microbenchmarks** - optimized kernels frequently found in ML pipelines<br>• **Primitives** - semantically atomic operations such as ciphertext-ciphertext multiplication and rotations |
| **Hardware Profiler** | One-command harness that captures runtime, micro-architectural counters, and power/energy on AMD & Intel CPUs under either standard or user-supplied security parameters                                                                                                                            |
| **Predictive Model**  | Lightweight performance model that extrapolates primitive-level measurements to full-workload runtime and energy with single-digit-percentage error, enabling rapid design-space exploration                                                                                   |

#### Key Highlights  

* 📊 **Reproducible evaluation** across workloads, parameter sets, and hardware vendors  
* ⚡ **Fast design-space exploration** via predictive models
* 🛠️ **Extensible backend support** - current release targets CPUs with OpenFHE; GPU back-ends and additional FHE libraries are on the roadmap  
* 🔍 **Rich metadata logging** - all security, algorithmic, and hardware settings plus raw and modeled results are stored in JSON/CSV for downstream analysis  
* 🤝 **Community-driven** - contributions of new benchmarks, hardware targets, and modeling techniques are welcome  

---  

CryptoOracle serves as a shared platform to accelerate collaborative progress in FHE applications, algorithms, software, and hardware. Clone the repo, run the benchmarks, and join the effort!  

### Clone the Repository  

```bash
# TODO: ANON REMOVAL
````

## Features

* **Benchmarking**: Analyze latency, power consumption, and throughput for FHE primitives, microbenchmarks, and end-to-end workloads.
* **Performance Profiling**: Generate FlameGraphs and conduct runtime analysis using `perf` tools.
* **Predictive Modeling**: Rapidly explore configuration spaces with additive or ML-based runtime and energy models.
* **Configurable Parameters**: Fine-tune CKKS encryption parameters (security standard, ring dimension, depth, batch size, and OpenMP thread count) via intuitive CLI arguments.

## System Dependencies

> **Note:** Commands assume you are already in the **CryptOracle** root directory.

### Required Packages

| Package                                  | Reason                           |
| ---------------------------------------- | -------------------------------- |
| **git**                                  | Version control                  |
| **cmake**                                | Build system generator           |
| **autoconf / libtool / build-essential** | Core compilation toolchain       |
| **libgoogle-perftools-dev**              | `tcmalloc` for performance       |
| **linux-tools-\***                       | Kernel tools (e.g., `perf`)      |
| **python3-dev / python3-pip**            | Python headers & package manager |
| **boost-all-dev**                        | Portable C++ libraries           |

### System Configuration

Enable privileged performance counters:

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

### Install System Dependencies

> **Heads-up:** WSL *will not work* (missing full `perf` support as of 11/10/24).

```bash
sudo apt-get update -y && sudo apt-get upgrade -y
sudo apt install -y git cmake autoconf build-essential libtool \
  libgoogle-perftools-dev python3-dev python3-pip libboost-all-dev \
  linux-tools-$(uname -r) linux-tools-common linux-tools-generic
```

### Library Path Configuration

If you encounter “error while loading shared libraries” when running the benchmarks, export the OpenFHE installation path:

```bash
LIB_DIR="$(pwd)/openfhe-development-install/lib"
export LD_LIBRARY_PATH="$LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Persist to ~/.bashrc if not already present
grep -qxF "export LD_LIBRARY_PATH=\"$LIB_DIR:\$LD_LIBRARY_PATH\"" ~/.bashrc || \
  echo "export LD_LIBRARY_PATH=\"$LIB_DIR:\$LD_LIBRARY_PATH\"" >> ~/.bashrc
```

## Python Dependencies

The core benchmarking scripts are pure-Python and require:

* **Python ≥ 3.8**
* `psutil`, `py-cpuinfo`, `GPUtil`, `dmidecode` - hardware discovery
* `pandas`, `numpy` - data manipulation
* `matplotlib` - visualization
* `pyyaml` - structured config files

### Install Python Dependencies

```bash
sudo apt install python3.12-venv -y
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install psutil py-cpuinfo GPUtil dmidecode pyyaml pandas numpy matplotlib
```

## Running Microbenchmarks

```bash
python3 benchmark-main.py --power-latency-analysis True \
                          --runtime-analysis True \
                          --build True
```

## Complete CLI Example (with defaults)

```bash
python3 benchmark-main.py \
  -s none \
  -n 13 \
  -b 12 \
  --depth 5 \
  --num-threads 0 \
  --compiler-optimizations True \
  --build True \
  --run-primitives True \
  --csv-name test \
  --verbose
```

## CLI Argument Reference

| **Arg**                         | **Default** | **Description**                                                              |
| ------------------------------- | ----------- | ---------------------------------------------------------------------------- |
| `-s, --security-standard-level` | `none`      | CKKS security level (`none`, `128c`, `192c`, `256c`, `128q`, `192q`, `256q`) |
| `-n, --ring-dimension`          | `13`        | CKKS ring dimension exponent → actual N = 2<sup>N</sup>                      |
| `-b, --batch-size`              | `12`        | CKKS batch size exponent (2<sup>b</sup>, must < N)                           |
| `-d, --depth`                   | `5`         | Multiplicative depth                                                         |
| `-t, --num-threads`             | `0`         | OpenMP thread count (`0` = all logical cores)                                |
| `-p, --run-primitives`          | `False`     | Run primitive-level benchmarks                                               |
| `-k, --run-microbenchmarks`     | `False`     | Run microbenchmarks                                                          |
| `-w, --run-workloads`           | `False`     | Run full workloads                                                           |
| `-r, --runtime-analysis`        | `True`      | Collect latency & power metrics                                              |
| `-e, --event-profiling`         | `True`      | Collect runtime metrics                                                      |
| `-f, --flamegraph-generation`   | `False`     | Generate FlameGraphs                                                         |
| `-b, --build`                   | `True`      | (Re)build project before running                                             |
| `-o, --compiler-optimizations`  | `True`      | Enable compiler optimizations                                                |
| `--cold-caching`                | `True`      | Cold-cache primitive profiling                                               |
| `-c, --csv-name`                | `""`        | Output file suffix (`<level>-results-<csv>.csv`)                             |
| `-g, --run-group`               | `False`     | Run benchmarks defined in a YAML group file                                  |
| `--fhe`                         | `False`     | Enable FHE mode (required for bootstrapping primitive)                       |
| `-v, --verbose`                 | `False`     | Verbose logging                                                              |
| `-h, --help`                    |             | Show help message                                                            |

## Further Documentation

* **OpenFHE Documentation** - [https://openfhe-development.readthedocs.io/en/latest/](https://openfhe-development.readthedocs.io/en/latest/)

## Authors

* Anonymous until conference submission results announced # TODO: ANON REMOVAL

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.
Licenses for all third-party dependencies are inherited.