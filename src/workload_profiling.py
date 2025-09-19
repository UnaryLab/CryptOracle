import argparse
import os
import shutil
import subprocess
from datetime import datetime
from typing import List, Dict
from src.globals import script_globals
import src.utils as utils
import src.performance_utils as perf_utils
from src.logging_utils import print_info, print_error, print_status, print_section_header, print_timestamp
from src.microbenchmark_profiling import build_and_install_polycircuit
import src.logging_utils as log
import re
import yaml
import zipfile
import subprocess

# Perform setup, runtime analysis, event profiling, flamegraph generation, and results recording for workloads
def analyze_workload_performance(args: argparse.Namespace) -> None:
    initialize_git_submodules(args)
    unzip_weights()
    validate_submodules()

    import_workloads()
    
    if args.build:
        print_status("Building workload benchmarking application...")
        build_workloads(args)

    if args.runtime_analysis:
        print_status("Performing runtime analysis for workloads...")
        runtime_analysis_start = datetime.now()
        runtime_analysis_workloads(args)
        runtime_analysis_end = datetime.now()
        print_timestamp(f"Runtime analysis completed in {(runtime_analysis_end - runtime_analysis_start).total_seconds():.3f} seconds.")
        
    if args.event_profiling:
        print_status("Performing event profiling for workloads...")
        event_profiling_start = datetime.now()
        workload_event_profiling(args)
        event_profiling_end = datetime.now()
        print_timestamp(f"Event profiling completed in {(event_profiling_end - event_profiling_start).total_seconds():.3f} seconds.")
    
    workload_save_csv(args)
    
    if args.flamegraph_generation:
        print_status("Generating FlameGraphs for workloads...")
        workload_generate_flamegraph(args)
        

def initialize_git_submodules(args: argparse.Namespace):
    submodules = {
        "benchmarks/contrib": "https://github.com/openfheorg/contrib.git",
        "benchmarks/openfhe-logreg-training-examples": "https://github.com/openfheorg/openfhe-logreg-training-examples",
        "benchmarks/openfhe-genomic-examples": "https://github.com/openfheorg/openfhe-genomic-examples",
    }
    
    for submodule, url in submodules.items():
        submodule_path = utils.get_absolute_path(submodule)
        if not os.path.isdir(submodule_path):
            log.print_info(f"Initializing submodule: {submodule}")
            try:
                subprocess.run(
                    ["git", "submodule", "update", "--init", "--recursive", submodule_path],
                    check=True,
                    stdout=None if args.verbose else subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
                log.print_info(f"Submodule {submodule} initialized successfully.")
            except subprocess.CalledProcessError as e:
                log.print_info(f"Submodule {submodule} not found, adding it: {submodule}")
                try:
                    subprocess.run(
                        ["git", "submodule", "add", "-f", url, submodule_path],
                        check=True,
                        stdout=None if args.verbose else subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                    )
                    subprocess.run(
                        ["git", "submodule", "update", "--init", "--recursive", submodule_path],
                        check=True,
                        stdout=None if args.verbose else subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                    )
                    log.print_info(f"Submodule {submodule} added and initialized successfully.")
                except subprocess.CalledProcessError as e:
                    log.print_error(f"Failed to add submodule {submodule}: {e.stderr.decode().strip()}")
        else:
            log.print_info(f"Updating existing submodule: {submodule}")
            try:
                subprocess.run(
                    ["git", "submodule", "update", "--init", "--recursive", submodule_path],
                    check=True,
                    stdout=None if args.verbose else subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
                log.print_info(f"Submodule {submodule} updated successfully.")
            except subprocess.CalledProcessError as e:
                log.print_error(f"Failed to update submodule {submodule}: {e.stderr.decode().strip()}")

    log.print_info("Git submodules initialized.")

# Unzip model weights for ResNet20
def unzip_weights():
    resnet_dir = utils.get_absolute_path("benchmarks/contrib/images-resnet20-low-mem/")
    weights_zip = os.path.join(resnet_dir, "weights.zip")
    weights_dir = os.path.join(resnet_dir, "weights")

    if os.path.exists(weights_zip) and not os.path.exists(weights_dir):
        print_info("Extracting weights.zip for ResNet20...")
        with zipfile.ZipFile(weights_zip, 'r') as zip_ref:
            zip_ref.extractall(resnet_dir)
    elif not os.path.exists(weights_zip):
        print_error(f"weights.zip not found at {weights_zip}")
    else:
        print_info("Weights already extracted.")

# Check to ensure required submodules are initialized
def validate_submodules():
    required_paths = [
        "benchmarks/contrib/images-resnet20-low-mem",
        "benchmarks/openfhe-logreg-training-examples",
        "benchmarks/openfhe-genomic-examples"
    ]
    for path in required_paths:
        abs_path = utils.get_absolute_path(path)
        if not os.path.isdir(abs_path) or not os.listdir(abs_path):
            print_error(f"Missing or empty submodule: {path}")
            raise FileNotFoundError(f"Submodule {path} is not initialized correctly.")

# Import the configured workloads for profiling
def import_workloads() -> None:
    """Imports workloads from a YAML file."""
    file_path = utils.get_absolute_path("in/workloads.yaml")

    try:
        with open(file_path, "r") as file:
            data = yaml.safe_load(file)
            script_globals.workloads = data.get("workloads", [])
    except Exception as e:
        print_error(f"Error loading {file_path}: {str(e)}")
        raise

# Builds OpenFHE workloads
def build_workloads(args: argparse.Namespace) -> None:
    build_and_install_polycircuit(args)
    build_low_memory_resnet20(args)
    build_logistic_regression(args)
    build_chi_square_test(args)
    
def build_low_memory_resnet20(args: argparse.Namespace) -> None:
    print_info("Building LowMemoryFHEResNet20 workload...")
    base_dir = utils.get_absolute_path("benchmarks/contrib/images-resnet20-low-mem")
    build_dir = os.path.join(base_dir, "build")
    
    replacement_files = ["FHEController.cpp", "main.cpp", "Utils.h"]
    replacement_files_dir = utils.get_absolute_path("replacement-files")

    for replacement_file in replacement_files:
        shutil.copy2(os.path.join(replacement_files_dir, "resnet", replacement_file), os.path.join(base_dir, "src"))

    os.makedirs(build_dir, exist_ok=True)

    # Patch crypto parameters BEFORE building
    # patch_resnet_params(base_dir)

    subprocess.run(['cmake', '-B', build_dir, '-S', base_dir], 
                  check=True,
                  stdout=None if args.verbose else subprocess.DEVNULL,
                  stderr=subprocess.PIPE)
    result = subprocess.run(
        ['cmake', '--build', build_dir, '--target', 'LowMemoryFHEResNet20'],
        check=True,
        stdout=None if args.verbose else subprocess.DEVNULL,
        stderr=subprocess.PIPE
    )
    if result.returncode != 0:
        print(result.stderr.decode())
        print_error("Building LowMemoryFHEResNet20 failed")

def patch_resnet_params(base_dir: str) -> None:
    fhe_file = os.path.join(base_dir, "src", "FHEController.cpp")

    if os.path.exists(fhe_file):
        with open(fhe_file, "r") as file:
            content = file.read()

        # Patch security level, depth, ring dim, scaling mod size
        content = re.sub(r"SetSecurityLevel\([^)]+\)", "SetSecurityLevel(HEStd_NotSet)", content)
        content = re.sub(r"SetMultiplicativeDepth\s*\([^\)]+\)", "SetMultiplicativeDepth(circuit_depth+1)", content)
        content = re.sub(r"SetRingDim\(\d+\)", "SetRingDim(1 << 16);\n    cc->GetCryptoParameters()->SetScalingModSize(30)", content)

        with open(fhe_file, "w") as file:
            file.write(content)

# Generate keys only if they don't already exist.
def generate_keys_clean(base_dir: str, args: argparse.Namespace) -> None:
    build_dir = os.path.join(base_dir, "build")
    keys_dir = os.path.join(base_dir, "keys_exp1")
    binary_path = os.path.join(build_dir, "LowMemoryFHEResNet20")

    if os.path.isdir(keys_dir) and os.listdir(keys_dir):
        print_info("Keys already exist — skipping key generation.")
        return

    print_info("Generating new keys...")

    try:
        subprocess.run(
            [binary_path, "generate_keys", "1", "verbose", "2"],
            cwd=build_dir,
            check=True,
            stdout=None if args.verbose else subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as e:
        print_error(f"Key generation failed:\n{e.stderr.decode()}")
        raise RuntimeError("Key generation failed.")

def build_logistic_regression(args: argparse.Namespace) -> None:
    print_info("Building Logistic Regression workload...")
    base_dir = utils.get_absolute_path("benchmarks/openfhe-logreg-training-examples")
    build_dir = os.path.join(base_dir, "build")
    replacement_files = ["lr_nag.cpp", "parameters.h"]
    replacement_files_dir = utils.get_absolute_path("replacement-files")

    for replacement_file in replacement_files:
        shutil.copy2(os.path.join(replacement_files_dir, "logistic_regression", replacement_file), os.path.join(base_dir, replacement_file))

    os.makedirs(build_dir, exist_ok=True)

    subprocess.run(
        ['cmake', '-B', build_dir, '-S', base_dir],
        check=True,
        stdout=None if args.verbose else subprocess.DEVNULL,
        stderr=subprocess.PIPE
    )

    # Build the correct target: lr_nag
    result = subprocess.run(
        ['cmake', '--build', build_dir, '--target', 'lr_nag'],
        check=True,
        stdout=None if args.verbose else subprocess.DEVNULL,
        stderr=subprocess.PIPE
    )

    if result.returncode != 0:
        print(result.stderr.decode())
        print_error("Building lr_nag failed")

def build_chi_square_test(args: argparse.Namespace) -> None:
    print_info("Building Chi-Square Test workload...")
    base_dir = utils.get_absolute_path("benchmarks/openfhe-genomic-examples")
    build_dir = os.path.join(base_dir, "build")
    os.makedirs(build_dir, exist_ok=True)

    replacement_files = ["demo-chi2.cpp"]
    replacement_files_dir = utils.get_absolute_path("replacement-files")

    for replacement_file in replacement_files:
        shutil.copy2(os.path.join(replacement_files_dir, "chi_square_test", replacement_file), os.path.join(base_dir, replacement_file))

    subprocess.run(['cmake', '-B', build_dir, '-S', base_dir],
                   check=True,
                   stdout=None if args.verbose else subprocess.DEVNULL,
                   stderr=subprocess.PIPE)

    result = subprocess.run(['cmake', '--build', build_dir, '--target', 'demo-chi2'],
                    check=True,
                    stdout=None if args.verbose else subprocess.DEVNULL,
                    stderr=subprocess.PIPE)

    if result.returncode != 0:
        print(result.stderr.decode())
        print_error("Building Chi-Square Test workload failed")

def run_cifar10(args: argparse.Namespace, setup: bool = False) -> None:
    polycircuit_binaries_path = utils.get_absolute_path("benchmarks/polycircuit/examples")
    input_image_path = utils.get_absolute_path("benchmarks/polycircuit/examples/CIFAR10Usage/class-1.txt")
    
    cmd = [
        os.path.join(polycircuit_binaries_path, "build", "CIFAR10Usage", "cifar"),
        "--input", input_image_path,
        "--setup", "true" if setup else "false"
    ]

    try:
        subprocess.run(cmd, check=True,
            stdout=None if args.verbose else subprocess.DEVNULL,
            stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(e.stderr.decode())
        print_error("CIFAR10 workload failed to run")
        
def run_low_memory_resnet20(args: argparse.Namespace, setup: bool = False) -> None:
    base_dir = utils.get_absolute_directory("benchmarks/contrib/images-resnet20-low-mem")
    build_dir = os.path.join(base_dir, "build")
    binary_path = os.path.join(build_dir, "LowMemoryFHEResNet20")
    
    # Always fresh keys
    generate_keys_clean(base_dir, args)

    input_dir = os.path.join(base_dir, "inputs")
    os.makedirs(input_dir, exist_ok=True)

    image_path = os.path.join(input_dir, "vale.jpg")

    if not os.path.isfile(image_path):
        print(f"WARNING: Default image not found at {image_path}. Checking for other images...")
        jpg_files = [f for f in os.listdir(input_dir) if f.endswith('.jpg')]
        if jpg_files:
            image_path = os.path.join(input_dir, jpg_files[0])
        else:
            return

    relative_image_path = os.path.relpath(image_path, start=os.path.dirname(build_dir))

    cmd = [
        binary_path,
        'load_keys', '1',
        'input', relative_image_path,
        'verbose', '2',
        'setup' if setup else ''
    ]

    try:
        subprocess.run(
            cmd,
            cwd=build_dir,
            check=True,
            stdout=None if args.verbose else subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Running LowMemoryFHEResNet20 failed:\n{e.stderr.decode()}")
        return

def run_logistic_regression(args: argparse.Namespace, setup: bool = False) -> None:
    base_dir = utils.get_absolute_directory("benchmarks/openfhe-logreg-training-examples")
    build_dir = os.path.join(base_dir, "build")
    
    # Change directory to build and run commands
    os.chdir(build_dir)

    lr_nag_command = ['./lr_nag', '-n', '10', '-s', '1' if setup else '0']
    result = subprocess.run(lr_nag_command, check=True,
                        stdout=None if args.verbose else subprocess.DEVNULL,
                        stderr=subprocess.PIPE)
    if result.returncode != 0:
        print(result.stderr.decode())
        print_error("LogReg failed to run")


def run_chi_square_test(args: argparse.Namespace, setup: bool = False) -> None:
    base_dir = utils.get_absolute_directory("benchmarks/openfhe-genomic-examples")
    build_dir = os.path.join(base_dir, "build")

    os.chdir(build_dir)

    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    result = subprocess.run([
        './demo-chi2',
        '--SNPdir', '../data',
        '--SNPfilename', 'random_sample',
        '--pvalue', 'pvalue.txt',
        '--runtime', 'result.txt',
        '--samplesize', '200',
        '--snps', '16384',
        '--setup', 'true' if setup else 'false'
        
    ], check=True,
        stdout=None if args.verbose else subprocess.DEVNULL,
        stderr=subprocess.PIPE
    )

    if result.returncode != 0:
        print(result.stderr.decode())
        print_error("Chi-Square Test workload failed to run")

def runtime_analysis_workloads(args: argparse.Namespace) -> None:
    print_info("Performing runtime analysis for workloads...")

    perf_out_path = utils.get_absolute_path("out/temp/perf_out.perf")
    
    # Define the mapping between microbenchmark strings and their associated functions
    workload_functions = {
        "cifar10": lambda: run_cifar10(args, setup),
        "low_memory_resnet20": lambda: run_low_memory_resnet20(args, setup),
        "logistic_regression": lambda: run_logistic_regression(args, setup),
        "chi_square_test": lambda: run_chi_square_test(args, setup)
    }
    
    setup = False # First pass of monitoring is to extract the setup + execution times
    
    for workload in script_globals.workloads:
        if workload not in workload_functions:
            print_error(f"Unknown workload: {workload}")
            continue
        
        print_info(f"Running {workload} workload...")
        
        perf_proc = perf_utils.start_perf(perf_out_path)
        workload_functions[workload]()
        perf_utils.stop_perf(perf_proc)
        
        # Run a separate Python script to parse timing and power data
        python_parsed_output = subprocess.check_output(
            ["python3", utils.get_absolute_path("util/parse_microbench_and_power.py")]
        ).decode()

        execution_time = utils.extract_float_value(python_parsed_output, "Average Time")
        energy = utils.extract_float_value(python_parsed_output, "Energy Usage")
        power = float(energy / execution_time if execution_time > 0 else float("NaN"))
        
        script_globals.workload_setup_and_execution_times[workload] = float(execution_time)
        script_globals.workload_setup_and_execution_energies[workload] = float(energy)
        
        print_info(
            f"{workload}: Time: {execution_time:.4f}s, "
            f"Energy: {energy:.2f}J, "
            f"Power: {power:.2f}W"
        )
        
    setup = True # Second pass of monitoring is to extract the setup times
        
    for workload in script_globals.workloads:
        if workload not in workload_functions:
            print_error(f"Unknown workload: {workload}")
            continue
        
        print_info(f"Running {workload} workload in setup mode...")
        
        perf_proc = perf_utils.start_perf(perf_out_path)
        workload_functions[workload]()
        perf_utils.stop_perf(perf_proc)
        
        # Run a separate Python script to parse timing and power data
        python_parsed_output = subprocess.check_output(
            ["python3", utils.get_absolute_path("util/parse_microbench_and_power.py")]
        ).decode()

        script_globals.workload_setup_times[workload] = utils.extract_float_value(
            python_parsed_output, "Average Time"
        )
        script_globals.workload_setup_energies[workload] = utils.extract_float_value(
            python_parsed_output, "Energy Usage"
        )
        execution_time = utils.extract_float_value(python_parsed_output, "Average Time")
        energy = utils.extract_float_value(python_parsed_output, "Energy Usage")
        power = float(energy / execution_time if execution_time > 0 else float("NaN"))
        
        script_globals.workload_setup_times[workload] = float(execution_time)
        script_globals.workload_setup_energies[workload] = float(energy)
        
        print_info(
            f"{workload}: Time: {execution_time:.4f}s, "
            f"Energy: {energy:.2f}J, "
            f"Power: {power:.2f}W"
        )

    calculate_timings_and_energies_workloads()
    
    print_section_header("Timing and power results for workloads")
    for workload in workload_functions.keys():
        if workload in script_globals.workloads:
            print_info(f"{workload}:")
            print_info(f"  Time: {script_globals.workload_execution_times[workload]:.3f} s")
            print_info(f"  Energy: {script_globals.workload_execution_energies[workload]:.3f} J")
            print_info(f"  Power: {script_globals.workload_execution_powers[workload]:.3f} W")

def calculate_timings_and_energies_workloads() -> None:
    """Calculates the execution times, energies, and power consumption for each primitive."""
    for workload in script_globals.workloads:
        execution_time = abs(script_globals.workload_setup_and_execution_times[workload] - script_globals.workload_setup_times[workload])
        energy = abs(script_globals.workload_setup_and_execution_energies[workload] - script_globals.workload_setup_energies[workload])
        
        script_globals.workload_execution_times[workload] = execution_time
        script_globals.workload_execution_energies[workload] = energy
        script_globals.workload_execution_powers[workload] = (
            (energy / execution_time) if execution_time > 0 else float("inf")
        )
                
    
# Generates a dictionary containing the executable calls and arguments for each workload.
def generate_command_dict(workload_dir_path: str, args: argparse.Namespace) -> Dict[str, str]:
    commands = {}
    
    all_workload_commands = {
        "low_memory_resnet20": [
            os.path.join(workload_dir_path, "contrib", "images-resnet20-low-mem", "build", "LowMemoryFHEResNet20"),
            "load_keys", "1", "input", "inputs/vale.jpg"],
        "logistic_regression": [
            os.path.join(workload_dir_path, "openfhe-logreg-training-examples", "build", "lr_nag"),
            "-n", "10",
        ],

        "chi_square_test": [
            os.path.join(workload_dir_path, "openfhe-genomic-examples", "build", "demo-chi2"),
            "--SNPdir", "../data",
            "--SNPfilename", "random_sample",
            "--pvalue", "pvalue.txt",
            "--runtime", "result.txt",
            "--samplesize", "200",
            "--snps", "16384"
        ],
        "cifar10": [
            os.path.join(workload_dir_path, "polycircuit", "examples", "build","CIFAR10Usage", "cifar"),
            "--input", utils.get_absolute_path("benchmarks/polycircuit/examples/CIFAR10Usage/class-1.txt")]
    }
    
    # Add only the workloads enabled in workloads.yaml to the dictionary
    for workload in all_workload_commands:
        if workload in script_globals.workloads:
            commands[workload] = ' '.join(all_workload_commands[workload])

    return commands

# Performs event profiling for each workload based on the events specified inside of in/perf_events.yaml
def workload_event_profiling(args: argparse.Namespace) -> None:
    workload_dir_path = utils.get_absolute_path("benchmarks")
    
    build_dirs = {
        "low_memory_resnet20": utils.get_absolute_path("benchmarks/contrib/images-resnet20-low-mem/build"),
        "logistic_regression": utils.get_absolute_path("benchmarks/openfhe-logreg-training-examples/build"),
        "chi_square_test": utils.get_absolute_path("benchmarks/openfhe-genomic-examples/build"),
        "cifar10": utils.get_absolute_path("benchmarks/polycircuit/examples/build/CIFAR10Usage")
    }

    for event in script_globals.perf_events:
        for suffix in ["execution", "rate"]:
            setattr(script_globals, f"workload_{event}_{suffix}", {})

    event_string: str = ",".join(script_globals.perf_events)
    command_dict = generate_command_dict(workload_dir_path, args)
    script_globals.workload_command_dict = command_dict
    
    for workload, command_string in command_dict.items():
        script_globals.workload_setup_perf_results[workload] = {}
        script_globals.workload_setup_and_execution_perf_results[workload] = {}
        script_globals.workload_perf_results[workload] = {}
        execute_runtime_profiling_workload(workload, build_dirs[workload], command_string, event_string, args, setup_only=False)
        execute_runtime_profiling_workload(workload, build_dirs[workload], command_string, event_string, args, setup_only=True)
        
        # Compute perf event count based on setup and execution times
        perf_utils.compute_execution_metrics(workload, script_globals.workload_execution_times[workload], 
            script_globals.workload_setup_perf_results[workload], 
            script_globals.workload_setup_and_execution_perf_results[workload], 
            script_globals.workload_perf_results[workload])
        
    print_section_header("Event profiling results for workloads")
    for workload in script_globals.workloads:
        perf_utils.print_runtime_results(workload, script_globals.workload_perf_results[workload])
    

def execute_runtime_profiling_workload(workload: str, binary_dir_path: str, command_string: str, event_string: str, args: argparse.Namespace, setup_only: bool = False) -> None:
    setup_string = "in setup mode" if setup_only else ""
    print_info(f"Running {workload} workload in {setup_string}...")
    
    # Append the setup flag based on the command line arguments of each workload
    match workload:
        case "logistic_regression":
            command_string = command_string + (" -s 1" if setup_only else "")
        case "cifar10":
            command_string = command_string + " --setup" + (" true" if setup_only else " false")
        case "chi_square_test":
            command_string = command_string + " --setup" + (" true" if setup_only else " false")
        case "low_memory_resnet20":
            command_string = command_string + (" setup" if setup_only else "")
        case _:
            print_error(f"Unknown workload: {workload}")
            return
    
    run_perf_workload(binary_dir_path, command_string, event_string, args)
    perf_utils.process_metrics(workload, (script_globals.workload_setup_perf_results[workload] if setup_only else script_globals.workload_setup_and_execution_perf_results[workload]))
    utils.check_metrics(workload, (script_globals.workload_setup_perf_results[workload] if setup_only else script_globals.workload_setup_and_execution_perf_results[workload]))
    
def run_perf_workload(binary_dir_path: str,command_string: str, event_string: str, args: argparse.Namespace) -> None:
    polling_frequency = 1000

    cmd: List[str] = [
        "perf",
        "record",
        "-o",
        utils.get_absolute_path(os.path.join("out", "temp", "perf.data")),
        "-F",
        str(int(polling_frequency)),
        "--event",
        event_string,
        "--",
    ] + command_string.split()

    try:
        subprocess.run(
            cmd,
            cwd=binary_dir_path,
            check=True,
            stdout=None if args.verbose else subprocess.DEVNULL,
            stderr=subprocess.PIPE
        )
    except subprocess.CalledProcessError as e:
        print_error(f"Perf record failed for {command_string}")
    
    output_file_path = utils.get_absolute_path(os.path.join("out", "temp", "perf_report_out.txt"))
    
    with open(output_file_path, "w") as outfile:
        subprocess.run(
            ["perf", "report", "--stdio", "-i", utils.get_absolute_path(os.path.join("out", "temp", "perf.data"))],
            check=True,
            stdout=outfile,
            stderr=outfile,
        )

def compute_execution_metrics(workload: str) -> None:
    for event in script_globals.perf_events:
        event_count = float(script_globals.workload_perf_results.get(workload, {}).get(event, 0))
        script_globals.workload_perf_results[workload][event] = event_count

    total_exec_time = float(script_globals.workload_execution_times.get(workload, 0))
    cpu_cycles = float(script_globals.workload_perf_results.get(workload, {}).get("cpu-cycles", 0))

    script_globals.workload_perf_results[workload]["memory_bandwidth"] = utils.compute_memory_bandwidth(
        total_exec_time,
        float(script_globals.workload_perf_results[workload].get("cache-misses", 0))
    )

    script_globals.workload_perf_results[workload]["ipc"] = utils.compute_ipc(
        float(script_globals.workload_perf_results[workload].get("instructions", 0)),
        cpu_cycles
    )

    if utils.check_fp_ops_enabled() is True:
        script_globals.workload_perf_results[workload]["flops"] = utils.compute_flops(
            total_exec_time,
            float(script_globals.workload_perf_results[workload].get("fp_ops_retired_by_type.scalar_all", 0)),
            float(script_globals.workload_perf_results[workload].get("fp_ops_retired_by_type.vector_all", 0)),
            float(script_globals.workload_perf_results[workload].get("fp_ops_retired_by_type.scalar_add", 0))
        )

def workload_save_csv(args: argparse.Namespace) -> None:
    suffix = f"-{args.csv_name}" if args.csv_name else ""
    csv_output_path = utils.get_absolute_path(f"out/csv/workload-results{suffix}.csv")
    perf_utils.save_results_csv(csv_output_path, args, 
        script_globals.workloads, 
        script_globals.workload_execution_times, 
        script_globals.workload_execution_energies, 
        script_globals.workload_execution_powers, 
        script_globals.workload_perf_results)

# Returns the working directory for a given workload
def get_cwd_workload(workload: str) -> str:
    cwd: str = utils.get_project_root()
    if workload == "low_memory_resnet20":
        cwd = utils.get_absolute_path("benchmarks/contrib/images-resnet20-low-mem/build")
    elif workload == "logistic_regression":
        cwd = utils.get_absolute_path("benchmarks/openfhe-logreg-training-examples/build")
    elif workload == "chi_square_test":
        cwd = os.path.join(cwd, "benchmarks/openfhe-genomic-examples/build")
    else:
        cwd = utils.get_project_root()

# Generate flamegraphs for each workload
def workload_generate_flamegraph(args: argparse.Namespace) -> None:
    for workload in script_globals.workloads:
        print_info(f"Generating FlameGraph for {workload}...")
        cwd = get_cwd_workload(workload)
        command_string = script_globals.workload_command_dict[workload]
        match workload:
            case "logistic_regression":
                command_string = command_string
            case "cifar10":
                command_string = command_string + " --setup false"
            case "chi_square_test":
                command_string = command_string + " --setup false"
            case "low_memory_resnet20":
                command_string = command_string
            case _:
                print_error(f"Unknown workload: {workload}")
                return
        
        perf_utils.generate_flamegraph(args, workload, command_string.split(), cwd)