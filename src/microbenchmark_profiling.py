import argparse
import os
import shutil
import yaml
import subprocess
from datetime import datetime
from typing import List, Dict
from src.globals import script_globals
import src.utils as utils
import src.performance_utils as perf_utils
from src.logging_utils import print_info, print_error, print_status, print_section_header, print_timestamp

    
def analyze_microbenchmark_performance(args: argparse.Namespace) -> None:
    import_microbenchmarks()
    
    if args.build:
        print_status("Building microbenchmark benchmarking application...")
        build_and_install_polycircuit(args)

    if args.power_latency_analysis:
        power_latency_analysis_start = datetime.now()
        print_status("Performing latency and power analysis for microbenchmarks...")
        monitor_timing_and_power_microbenchmarks(args)
        power_latency_analysis_end = datetime.now()
        print_timestamp(f"Microbenchmark runtime analysis completed in {(power_latency_analysis_end - power_latency_analysis_start).total_seconds():.3f} seconds.")
        
    if args.runtime_analysis:
        print_status("Performing runtime analysis for microbenchmarks...")
        event_profiling_start = datetime.now()
        microbenchmark_runtime_analysis(args)
        event_profiling_end = datetime.now()
        print_timestamp(f"Microbenchmark event profiling completed in {(event_profiling_end - event_profiling_start).total_seconds():.3f} seconds.")
    
    if args.flamegraph_generation:
        print_status("Generating FlameGraphs for microbenchmarks...")
        microbenchmark_generate_flamegraph(args)

    microbenchmark_save_csv(args)
    
def import_microbenchmarks() -> None:
    """Imports microbenchmarks from a YAML file."""
    file_path = utils.get_absolute_path("in/microbenchmarks.yaml")

    try:
        with open(file_path, "r") as file:
            data = yaml.safe_load(file)
            script_globals.microbenchmarks = data.get("microbenchmarks", [])
    except Exception as e:
        print_error(f"Error loading {file_path}: {str(e)}")
        raise
    
def build_and_install_polycircuit(args: argparse.Namespace) -> None:
    """
    Clones the polycircuit repository, installs it, replaces the CMakeLists.txt,
    and builds the examples.
    """
    project_root = utils.get_project_root()
    polycircuit_repo_dir = os.path.join(project_root, "benchmarks", "polycircuit")
    polycircuit_install_dir = os.path.join(project_root, "benchmarks","polycircuit-install")
    examples_dir = os.path.join(polycircuit_repo_dir, "examples")
    replacement_files_dir = os.path.join(project_root, "replacement-files")
    cmake_replacement_file = os.path.join(replacement_files_dir, "CMakeLists.txt")
    replacement_files = {
        "MatrixMultiplication": ["MatrixMultiplicationUsage.cc", "MatrixMultiplication.hpp"],
        "SignEvaluation": ["SignEvaluationUsage.cc", "SignEvaluation.hpp"],
        "LogisticFunction": ["LogisticFunctionUsage.cc", "LogisticFunction.hpp"],
        "CIFAR10": ["CIFAR10ImageClassification.hpp", "cifar10.cc"]
    }
    target_cmakelists = os.path.join(examples_dir, "CMakeLists.txt")
    examples_build_dir = os.path.join(examples_dir, "build")
    openfhe_install_dir = os.path.join(project_root, "openfhe-development-install")
    
    print_info(f"Starting polycircuit setup process...")
    
    if not os.path.exists(polycircuit_repo_dir):
        print_info(f"Cloning polycircuit repository to {polycircuit_repo_dir}...")
        subprocess.run(
            ["git", "clone", "https://github.com/fairmath/polycircuit.git", polycircuit_repo_dir],
            check=True,
            stdout=None if args.verbose else subprocess.DEVNULL, 
            stderr=subprocess.PIPE
        )
    else:
        print_info(f"Polycircuit repository already exists at {polycircuit_repo_dir}")
    
    os.makedirs(polycircuit_install_dir, exist_ok=True)
    
    for replacement_file_dir_name, replacement_file_names in replacement_files.items():
        for replacement_file_name in replacement_file_names:
            if ".hpp" in replacement_file_name:
                target_dir_path = os.path.join(polycircuit_repo_dir, "include", "polycircuit", "component", replacement_file_dir_name)
                try:
                    target_dir_path = target_dir_path if replacement_file_dir_name != "CIFAR10" else target_dir_path + "ImageClassification"
                    print_info(f"Copying {replacement_file_name} to {target_dir_path}...")
                    shutil.copy2(os.path.join(replacement_files_dir, replacement_file_dir_name, replacement_file_name), target_dir_path)
                except FileNotFoundError:
                    print_info(f"Warning: Replacement file {replacement_file_name} not found.")
                    print_info("Skipping CMakeLists.txt replacement.")
            elif "Usage" in replacement_file_name:
                target_dir_path = os.path.join(polycircuit_repo_dir, "examples", replacement_file_dir_name + "Usage")
                try:
                    print_info(f"Copying {replacement_file_name} to {target_dir_path}...")
                    shutil.copy2(os.path.join(replacement_files_dir, replacement_file_dir_name, replacement_file_name), target_dir_path)
                except FileNotFoundError:
                    print_info(f"Warning: Replacement file {replacement_file_name} not found.")
                    print_info("Skipping CMakeLists.txt replacement.")
            elif "cifar10" in replacement_file_name:
                target_dir_path = os.path.join(polycircuit_repo_dir, "examples", "CIFAR10Usage")
                try:
                    print_info(f"Copying {replacement_file_name} to {target_dir_path}...")
                    shutil.copy2(os.path.join(replacement_files_dir, replacement_file_dir_name, replacement_file_name), target_dir_path)
                except FileNotFoundError:
                    print_info(f"Warning: Replacement file {replacement_file_name} not found.")
                    print_info("Skipping CMakeLists.txt replacement.")
        
    print_info(f"Building and installing polycircuit to {polycircuit_install_dir}...")
    polycircuit_build_dir = os.path.join(polycircuit_repo_dir, "build")
    os.makedirs(polycircuit_build_dir, exist_ok=True)
    
    try:
        subprocess.run(
            [
                "cmake", 
                "-B", polycircuit_build_dir,
                "-DCMAKE_INSTALL_PREFIX=" + polycircuit_install_dir,
                f"-DCMAKE_PREFIX_PATH={openfhe_install_dir}",
                polycircuit_repo_dir
            ],
            check=True,
            stdout=None if args.verbose else subprocess.DEVNULL, 
            stderr=subprocess.PIPE
        )
        
        subprocess.run(
            ["cmake", "--build", polycircuit_build_dir, "--target", "install", f"-j{script_globals.cpu_count}"],
            check=True,
            stdout=None if args.verbose else subprocess.DEVNULL, 
            stderr=subprocess.PIPE
        )
    except subprocess.CalledProcessError as e:
        print_error(f"Error during polycircuit build/install: {e}")
        return
    
    os.makedirs(examples_dir, exist_ok=True)
    
    print_info(f"Copying replacement CMakeLists.txt to {target_cmakelists}...")
    try:
        shutil.copy2(cmake_replacement_file, target_cmakelists)
    except FileNotFoundError:
        print_info(f"Warning: Replacement file {cmake_replacement_file} not found.")
        print_info("Skipping CMakeLists.txt replacement.")
    
    print_info(f"Building examples in {examples_build_dir}...")
    os.makedirs(examples_build_dir, exist_ok=True)
    
    try:
        subprocess.run(
            [
                "cmake",
                f"-DCMAKE_PREFIX_PATH={polycircuit_install_dir};{openfhe_install_dir}",
                ".."
            ],
            cwd=examples_build_dir,
            check=True,
            stdout=None if args.verbose else subprocess.DEVNULL, 
            stderr=subprocess.PIPE
        )
        
        subprocess.run(
            ["make", f"-j{script_globals.cpu_count}"],
            cwd=examples_build_dir,
            check=True,
            stdout=None if args.verbose else subprocess.DEVNULL, 
            stderr=subprocess.PIPE
        )
        
        print_info("Build completed successfully!")
    except subprocess.CalledProcessError as e:
        print_error(f"Error during examples build: {e}")
        return


def execute_polycircuit_command(cmd: List[str], args: argparse.Namespace) -> None:
    """Executes a Polycircuit command and handles errors."""
    try:
        subprocess.run(
            cmd, check=True, stdout=None if args.verbose else subprocess.DEVNULL, stderr=subprocess.PIPE
        )
    except subprocess.CalledProcessError:
        print_error(f"Run failed for {cmd}")


def run_matrix_multiplication(polycircuit_binaries_path: str, serialized_files_path: str, matrix_size: str, args: argparse.Namespace, setup: bool) -> None:
    """Runs the Matrix Multiplication microbenchmark."""
    cmd = [
        os.path.join(polycircuit_binaries_path, "build","MatrixMultiplicationUsage", "MatrixMultiplicationUsage"),
        "--size", str(args.matrix_size),
        "--cryptocontext_location", os.path.join(serialized_files_path, "cryptocontext.txt"),
        "--mult_key_location", os.path.join(serialized_files_path, "key_mult.txt"),
        "--rotate_key_location", os.path.join(serialized_files_path, "key_rot.txt"),
        "--a_input_ciphertext_location", os.path.join(serialized_files_path, "a_input_ciphertext.txt"),
        "--b_input_ciphertext_location", os.path.join(serialized_files_path, "b_input_ciphertext.txt"),
        "--output_ciphertext_location", os.path.join(serialized_files_path, "output_ciphertext.txt"),
        "--setup", "true" if setup else "false",
    ]
    execute_polycircuit_command(cmd, args)
    pass


def run_sign_eval(polycircuit_binaries_path: str, serialized_files_path: str, args: argparse.Namespace, setup: bool) -> None:
    """Runs the Sign Evaluation microbenchmark."""
    cmd = [
        os.path.join(polycircuit_binaries_path, "build","SignEvaluationUsage", "SignEvaluationUsage"),
        "--cryptocontext_location", os.path.join(serialized_files_path, "cryptocontext.txt"),
        "--mult_key_location", os.path.join(serialized_files_path, "key_mult.txt"),
        "--input_ciphertext_location", os.path.join(serialized_files_path, "a_input_ciphertext.txt"),
        "--output_ciphertext_location", os.path.join(serialized_files_path, "output_ciphertext.txt"),
        "--setup", "true" if setup else "false",
    ]
    execute_polycircuit_command(cmd, args)


def run_logistic_function(polycircuit_binaries_path: str, serialized_files_path: str, args: argparse.Namespace, setup: bool) -> None:
    """Runs the Logistic Function microbenchmark."""
    cmd = [
        os.path.join(polycircuit_binaries_path, "build", "LogisticFunctionUsage", "LogisticFunctionUsage"),
        "--cryptocontext_location", os.path.join(serialized_files_path, "cryptocontext.txt"),
        "--mult_key_location", os.path.join(serialized_files_path, "key_mult.txt"),
        "--input_ciphertext_location", os.path.join(serialized_files_path, "a_input_ciphertext.txt"),
        "--output_ciphertext_location", os.path.join(serialized_files_path, "output_ciphertext.txt"),
        "--setup", "true" if setup else "false",
    ]
    execute_polycircuit_command(cmd, args)


def monitor_timing_and_power_microbenchmarks(args: argparse.Namespace) -> None:
    """Monitors execution time, energy, and power usage for Polycircuit microbenchmarks."""
    polycircuit_binaries_path = utils.get_absolute_path("benchmarks/polycircuit/examples")
    serialized_files_path = utils.get_absolute_path("util/cryptocontext-generator/serialized-files")
    input_image_path = utils.get_absolute_path("benchmarks/polycircuit/examples/CIFAR10Usage/class-1.txt")

    perf_out_path = utils.get_absolute_path("out/temp/perf_out.perf")

    
    # Define the mapping between microbenchmark strings and their functions
    microbenchmark_functions = {
        "matrix_multiplication": lambda: run_matrix_multiplication(polycircuit_binaries_path, serialized_files_path, "32", args, setup),
        "sign_eval": lambda: run_sign_eval(polycircuit_binaries_path, serialized_files_path, args, setup),
        "logistic_function": lambda: run_logistic_function(polycircuit_binaries_path, serialized_files_path, args, setup),
    }

    setup = False # First pass of monitoring is to extract the setup + execution times
    
    for microbenchmark in script_globals.microbenchmarks:
        if microbenchmark not in microbenchmark_functions:
            print_error(f"Unknown microbenchmark: {microbenchmark}")
            continue
        
        print_info(f"Running {microbenchmark} microbenchmark...")

        perf_proc = perf_utils.start_perf(perf_out_path)
        microbenchmark_functions[microbenchmark]()
        perf_utils.stop_perf(perf_proc)

        # Run a separatePython script to parse timing and power data
        python_parsed_output = subprocess.check_output(["python3", utils.get_absolute_path("util/parse_microbench_and_power.py")]).decode()

        execution_time = utils.extract_float_value(python_parsed_output, "Average Time")
        energy = utils.extract_float_value(python_parsed_output, "Energy Usage")
        power = float(energy / execution_time if execution_time > 0 else float("NaN"))
        
        script_globals.microbenchmark_setup_and_execution_times[microbenchmark] = float(execution_time)
        script_globals.microbenchmark_setup_and_execution_energies[microbenchmark] = float(energy)
        
        print_info(
            f"{microbenchmark}: Time: {execution_time:.4f}s, "
            f"Energy: {energy:.2f}J, "
            f"Power: {power:.2f}W"
        )

    setup = True
    
    for microbenchmark in script_globals.microbenchmarks:
        if microbenchmark not in microbenchmark_functions:
            print_error(f"Unknown microbenchmark: {microbenchmark}")
            continue
        
        print_info(f"Running {microbenchmark} microbenchmark in setup mode...")

        perf_proc = perf_utils.start_perf(perf_out_path)
        microbenchmark_functions[microbenchmark]()
        perf_utils.stop_perf(perf_proc)
        
        # Run a separate Python script to parse timing and power data
        python_parsed_output = subprocess.check_output(["python3", utils.get_absolute_path("util/parse_microbench_and_power.py")]).decode()

        execution_time = utils.extract_float_value(python_parsed_output, "Average Time")
        energy = utils.extract_float_value(python_parsed_output, "Energy Usage")
        power = float(energy / execution_time if execution_time > 0 else float("NaN"))
        
        script_globals.microbenchmark_setup_times[microbenchmark] = float(execution_time)
        script_globals.microbenchmark_setup_energies[microbenchmark] = float(energy)
        
        print_info(
            f"{microbenchmark}: Time: {execution_time:.4f}s, "
            f"Energy: {energy:.2f}J, "
            f"Power: {power:.2f}W"
        )
    
    calculate_timings_and_energies_microbenchmarks()
    
    print_section_header("Timing and power results for microbenchmarks")
    for microbenchmark in microbenchmark_functions.keys():
        if microbenchmark in script_globals.microbenchmarks:
            print_info(f"{microbenchmark}:")
            print_info(f"  Time: {script_globals.microbenchmark_execution_times[microbenchmark]:.3f} s")
            print_info(f"  Energy: {script_globals.microbenchmark_execution_energies[microbenchmark]:.3f} J")
            print_info(f"  Power: {script_globals.microbenchmark_execution_powers[microbenchmark]:.3f} W")


def calculate_timings_and_energies_microbenchmarks() -> None:
    """Calculates the execution times, energies, and power consumption for each primitive."""
    for microbenchmark in script_globals.microbenchmarks:
        execution_time = abs(script_globals.microbenchmark_setup_and_execution_times[microbenchmark] - script_globals.microbenchmark_setup_times[microbenchmark])
        energy = abs(script_globals.microbenchmark_setup_and_execution_energies[microbenchmark] - script_globals.microbenchmark_setup_energies[microbenchmark])
        
        script_globals.microbenchmark_execution_times[microbenchmark] = execution_time
        script_globals.microbenchmark_execution_energies[microbenchmark] = energy
        script_globals.microbenchmark_execution_powers[microbenchmark] = (
            (energy / execution_time) if execution_time > 0 else float("inf")
        )
        
def generate_command_dict(polycircuit_binaries_path: str, serialized_files_path: str, input_image_path: str, args: argparse.Namespace) -> Dict[str, str]:
    commands = {}

    all_microbenchmark_commands = {
        "matrix_multiplication": [
            os.path.join(polycircuit_binaries_path, "build","MatrixMultiplicationUsage", "MatrixMultiplicationUsage"),
            "--size", str(args.matrix_size),
            "--cryptocontext_location", os.path.join(serialized_files_path, "cryptocontext.txt"),
            "--mult_key_location", os.path.join(serialized_files_path, "key_mult.txt"),
            "--rotate_key_location", os.path.join(serialized_files_path, "key_rot.txt"),
            "--a_input_ciphertext_location", os.path.join(serialized_files_path, "a_input_ciphertext.txt"),
            "--b_input_ciphertext_location", os.path.join(serialized_files_path, "b_input_ciphertext.txt"),
            "--output_ciphertext_location", os.path.join(serialized_files_path, "output_ciphertext.txt"),
        ],
        "sign_eval": [
            os.path.join(polycircuit_binaries_path, "build","SignEvaluationUsage", "SignEvaluationUsage"),
            "--cryptocontext_location", os.path.join(serialized_files_path, "cryptocontext.txt"),
            "--mult_key_location", os.path.join(serialized_files_path, "key_mult.txt"),
            "--input_ciphertext_location", os.path.join(serialized_files_path, "a_input_ciphertext.txt"),
            "--output_ciphertext_location", os.path.join(serialized_files_path, "output_ciphertext.txt"),
        ],
        "logistic_function": [
            os.path.join(polycircuit_binaries_path, "build","LogisticFunctionUsage", "LogisticFunctionUsage"),
            "--cryptocontext_location", os.path.join(serialized_files_path, "cryptocontext.txt"),
            "--mult_key_location", os.path.join(serialized_files_path, "key_mult.txt"),
            "--input_ciphertext_location", os.path.join(serialized_files_path, "a_input_ciphertext.txt"),
            "--output_ciphertext_location", os.path.join(serialized_files_path, "output_ciphertext.txt"),
        ],
    }

    for microbenchmark in script_globals.microbenchmarks:
        if microbenchmark in all_microbenchmark_commands:
            commands[microbenchmark] = ' '.join(all_microbenchmark_commands[microbenchmark])
        else:
            print_error(f"Unknown microbenchmark: {microbenchmark}")

    return commands

    
def microbenchmark_runtime_analysis(args: argparse.Namespace) -> None:
    """Runs runtime analysis for microbenchmarks."""
    polycircuit_binaries_path = utils.get_absolute_path("benchmarks/polycircuit/examples")
    serialized_files_path = utils.get_absolute_path("util/cryptocontext-generator/serialized-files")
    input_image_path = utils.get_absolute_path("benchmarks/polycircuit/examples/CIFAR10Usage/class-1.txt")

    for event in script_globals.perf_events:
        for suffix in ["execution", "rate"]:
            setattr(script_globals, f"microbenchmark_{event}_{suffix}", {})

    event_string: str = ",".join(script_globals.perf_events)
    command_dict = generate_command_dict(polycircuit_binaries_path, serialized_files_path, input_image_path, args)
    script_globals.microbenchmark_command_dict = command_dict
    for microbenchmark, command_string in command_dict.items():
        script_globals.microbenchmark_setup_perf_results[microbenchmark] = {}
        script_globals.microbenchmark_setup_and_execution_perf_results[microbenchmark] = {}
        script_globals.microbenchmark_perf_results[microbenchmark] = {}
        execute_runtime_profiling_microbenchmark(microbenchmark, command_string, event_string, args, setup_only=False)
        execute_runtime_profiling_microbenchmark(microbenchmark, command_string, event_string, args, setup_only=True)
        
        # Compute perf event count based on setup and execution times
        perf_utils.compute_execution_metrics(microbenchmark, script_globals.microbenchmark_execution_times[microbenchmark], 
            script_globals.microbenchmark_setup_perf_results[microbenchmark], 
            script_globals.microbenchmark_setup_and_execution_perf_results[microbenchmark], 
            script_globals.microbenchmark_perf_results[microbenchmark])
        
    print_section_header("Runtime analysis results for microbenchmarks")
    for microbenchmark in script_globals.microbenchmarks:
        perf_utils.print_runtime_results(microbenchmark, script_globals.microbenchmark_perf_results[microbenchmark])
        

def execute_runtime_profiling_microbenchmark(microbenchmark: str, command_string: str, event_string: str, args: argparse.Namespace, setup_only: bool = False) -> None:
    """Performs runtime analysis for a single microbenchmark."""
    setup_string = "in setup mode" if setup_only else ""
    print_info(f"Running {microbenchmark} microbenchmark in {setup_string}...")
    command_string = command_string + " --setup" + (" true" if setup_only else " false")
    
    run_perf_microbenchmark(command_string, event_string, args)
    perf_utils.process_metrics(microbenchmark, (script_globals.microbenchmark_setup_perf_results[microbenchmark] if setup_only else script_globals.microbenchmark_setup_and_execution_perf_results[microbenchmark]))
    utils.check_metrics(microbenchmark, (script_globals.microbenchmark_setup_perf_results[microbenchmark] if setup_only else script_globals.microbenchmark_setup_and_execution_perf_results[microbenchmark]))
    

def run_perf_microbenchmark(
    command_string: str, event_string: str, args: argparse.Namespace
) -> None:
    """Runs the perf tool to collect performance data."""
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
            check=True,
            stdout=None if args.verbose else subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError:
        print_error(f"Perf record failed for {command_string}")

    output_file_path = utils.get_absolute_path(os.path.join("out", "temp", "perf_report_out.txt"))

    with open(output_file_path, "w") as outfile:
        subprocess.run(
            ["perf", "report", "--stdio", "-i", utils.get_absolute_path(os.path.join("out", "temp", "perf.data"))],
            check=True,
            stdout=outfile,
            stderr=outfile,
        )

def microbenchmark_save_csv(args: argparse.Namespace) -> None:
    suffix = f"-{args.csv_name}" if args.csv_name else ""
    csv_output_path = utils.get_absolute_path(f"out/csv/microbenchmark-results{suffix}.csv")
    perf_utils.save_results_csv(csv_output_path, args, 
        script_globals.microbenchmarks, 
        script_globals.microbenchmark_execution_times, 
        script_globals.microbenchmark_execution_energies, 
        script_globals.microbenchmark_execution_powers, 
        script_globals.microbenchmark_perf_results)
         
def microbenchmark_generate_flamegraph(args: argparse.Namespace) -> None:
    cwd = utils.get_project_root()
    
    for microbenchmark in script_globals.microbenchmarks:
        print_info(f"Generating FlameGraph for {microbenchmark}...")
        command_string = script_globals.microbenchmark_command_dict[microbenchmark]
        match microbenchmark:
            case "matrix_multiplication":
                command_string = command_string + " --setup false"
            case "sign_eval":
                command_string = command_string + " --setup false"
            case "logistic_function":
                command_string = command_string + " --setup false"
        perf_utils.generate_flamegraph(args, microbenchmark, command_string.split(), cwd)
    