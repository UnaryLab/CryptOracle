import argparse
import subprocess
import os
from typing import List
import src.utils as utils
from src.logging_utils import print_info, print_error, print_status, print_section_header
import src.performance_utils as perf_utils
import csv
import re
import shutil
from src.globals import script_globals
import yaml

def analyze_primitive_performance(args: argparse.Namespace) -> None:
    """Analyzes the application performance and generates FlameGraphs if requested."""
    import_primitives(args)
    
    if args.build:
        print_status("Building primitive benchmarking application...")
        setup_and_build_primitive_app(args)
        
    if args.power_latency_analysis:
        print_status("Performing latency and power analysis for primitives...")
        monitor_timing_and_power_primitives(args)

    if args.runtime_analysis:
        print_status("Performing runtime analysis for primitives...")
        runtime_analysis(args)

    primitive_save_csv(args)
    
    if args.flamegraph_generation:
        print_status("Generating FlameGraphs for primitives...")
        primitive_generate_flamegraph(args)
    
def import_primitives(args: argparse.Namespace) -> None:
    """Imports input functions from a YAML file and updates command-line arguments."""
    file_path = utils.get_absolute_path("in/primitives.yaml")

    try:
        with open(file_path, "r") as file:
            data = yaml.safe_load(file)
            for key, value in data.items():
                setattr(args, key, value)
                if "CKKS_Bootstrap" in value and args.fhe is False:
                    print_error("CKKS_Bootstrap requires FHE mode. Please set --fhe to True.")
                    raise ValueError("FHE mode must be enabled for CKKS_Bootstrap.")
    except Exception as e:
        print_error(f"Error loading {file_path}: {str(e)}")
        raise
    
def prepare_primitive_app() -> None:
    """Prepare the OpenFHE benchmarking application."""
    target_cmakelists = utils.get_absolute_path(os.path.join("benchmarks", "ckks-primitives-serialized", "CMakeLists.txt"))
    
    if not os.path.exists(target_cmakelists):
        source_cmakelists = utils.get_absolute_path(os.path.join("openfhe-development", "CMakeLists.User.txt"))
        try:
            shutil.copy(source_cmakelists, target_cmakelists)
            with open(target_cmakelists, "a") as f:
                f.write(
                    "add_executable(ckks-primitives-application ckks-primitives.cpp)\n"
                )
            print_info("Created new CMakeLists.txt for primitive app")
        except OSError as e:
            print_error(f"Failed to prepare primitive app: {str(e)}")
    else:
        print_info("Using existing CMakeLists.txt for primitive app")


def setup_and_build_primitive_app(args: argparse.Namespace) -> None:
    """Setup and build the OpenFHE benchmarking application."""

    benchmark_dir = utils.get_absolute_path(os.path.join("benchmarks", "ckks-primitives-serialized"))
    if not os.path.exists(benchmark_dir):
        print_error(f"Primitive benchmarking application directory not found at: {benchmark_dir}")
        return
        
    build_dir = os.path.join(benchmark_dir, "build")
    os.makedirs(build_dir, exist_ok=True)

    print_info("Running CMake for OpenFHE primitive benchmarking application...")
    result = subprocess.run(
        ["cmake", *script_globals.cmake_opts, ".."],
        cwd=build_dir,
        stdout=None if args.verbose else subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    print() if args.verbose else None
    if result.returncode != 0:
        print(result.stderr.decode())
        print_error("CMake configuration for OpenFHE primitive benchmarking application failed")
        return

    print_info(
        "Running Make for OpenFHE primitive benchmarking application (can take a while)..."
    )
    result = subprocess.run(
        ["make", f"-j{script_globals.cpu_count}"],
        cwd=build_dir,
        stdout=None if args.verbose else subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    print() if args.verbose else None
    if result.returncode != 0:
        print(result.stderr.decode())
        print_error("Make for OpenFHE primitive benchmarking application failed")

def monitor_timing_and_power_primitives(args: argparse.Namespace) -> None:
    """Monitors timing and power usage during benchmark execution."""
    setup = False
    function_repeats = 0

    for primitive in args.primitives:
        print_info(f"Running primitive {primitive}...")

        perf_out_path = utils.get_absolute_path(os.path.join("out", "temp", "perf_out.perf"))
        microbench_out_path = utils.get_absolute_path(os.path.join("out", "temp", "microbench_out.txt"))

        run_primitive(primitive, setup, function_repeats, args, microbench_out_path, perf_out_path)

        script_globals.microbenchmarking_application_output = utils.read_file(microbench_out_path)

        python_parsed_output = subprocess.check_output(
            ["python3", utils.get_absolute_path(os.path.join("util", "parse_microbench_and_power.py"))]
        ).decode()

        script_globals.setup_and_execution_average_time[primitive] = utils.extract_float_value(
            python_parsed_output, "Average Time"
        )
        script_globals.setup_and_execution_total_energy[primitive] = utils.extract_float_value(
            python_parsed_output, "Energy Usage"
        )
        script_globals.setup_and_execution_num_calls[primitive] = utils.extract_int_value(
            script_globals.microbenchmarking_application_output, "was run"
        )

        print_info(
            f"{primitive}: Time: {script_globals.setup_and_execution_average_time[primitive]:.4f}s, "
            f"Energy: {script_globals.setup_and_execution_total_energy[primitive]:.2f}J, "
            f"Calls: {script_globals.setup_and_execution_num_calls[primitive]}"
        )

    setup = True
    print_info("Running setup-only phase...")

    for primitive in args.primitives:
        print_info(f"Running setup for {primitive}...")

        function_repeats = script_globals.setup_and_execution_num_calls[primitive]

        perf_out_path = utils.get_absolute_path(os.path.join("out", "temp", "perf_out.perf"))
        microbench_out_path = utils.get_absolute_path(os.path.join("out", "temp", "microbench_out.txt"))

        run_primitive(primitive, setup, function_repeats, args, microbench_out_path, perf_out_path)

        script_globals.microbenchmarking_application_output = utils.read_file(microbench_out_path)

        python_parsed_output = subprocess.check_output(
            ["python3", utils.get_absolute_path(os.path.join("util", "parse_microbench_and_power.py"))]
        ).decode()

        script_globals.setup_average_time[primitive] = utils.extract_float_value(python_parsed_output, "Average Time")
        script_globals.setup_total_energy[primitive] = utils.extract_float_value(python_parsed_output, "Energy Usage")
        script_globals.setup_num_calls[primitive] = utils.extract_int_value(script_globals.microbenchmarking_application_output, "was run")
        
        print_info(
            f"{primitive}: Time: {script_globals.setup_average_time[primitive]:.4f}s, "
            f"Energy: {script_globals.setup_total_energy[primitive]:.2f}J, "
            f"Calls: {script_globals.setup_num_calls[primitive]}"
        )
        
    calculate_timings_and_energies(args)
    
    print_section_header("Timing and Power Results")
    for primitive in args.primitives:
        print_info(f"{primitive}:")
        print_info(f"  Time: {script_globals.primitive_execution_times[primitive]:.6f} s")
        print_info(f"  Energy: {script_globals.primitive_execution_energies[primitive]:.6f} J")
        print_info(f"  Power: {script_globals.primitive_execution_powers[primitive]:.6f} W")


def run_primitive(
    primitive: str,
    setup: bool,
    function_repeats: int,
    args: argparse.Namespace,
    output_path: str,
    perf_out_path: str,
) -> None:
    """Runs a CKKS primitive benchmark with given parameters."""
    cmd = [
        utils.get_absolute_path(os.path.join("benchmarks", "ckks-primitives-serialized", "build", "ckks-primitives-application")),
        primitive,
        str(setup),
        str(args.cold_caching),
        str(function_repeats),
    ]

    with open(output_path, "w") as output_file:
        # if args.verbose:
        #     perf_proc = start_perf(perf_out_path)
        #     process = subprocess.Popen(
        #         cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=utils.get_project_root())
        #     stop_perf(perf_proc)
        #     for line in process.stdout:
        #         print(line, end="")
        #         output_file.write(line)

        #     process.stdout.close()
        #     result = process.wait()

        #     if result != 0:
        #         print(process.stderr.read())

        # else:
        perf_proc = perf_utils.start_perf(perf_out_path)
        result = subprocess.run(cmd, stdout=output_file, stderr=subprocess.PIPE, cwd=utils.get_project_root())
        perf_utils.stop_perf(perf_proc)
        if result.returncode != 0:
            print(result.stderr.decode())
            print_error("CKKS primitive execution failed")
                
def runtime_analysis(args: argparse.Namespace) -> None:
    """Conducts runtime analysis for CKKS microbenchmarks."""
    for event in script_globals.perf_events:
        for suffix in ["setup", "setup_and_execution", "execution", "rate", "cycle_rate"]:
            setattr(script_globals, f"{event}_{suffix}", {})

    event_string: str = ",".join(script_globals.perf_events)
    for primitive in args.primitives:
        print_info(f"Performing runtime analysis for {primitive}...")
        execute_runtime_profiling(primitive, event_string, args, setup_only=False)
        execute_runtime_profiling(primitive, event_string, args, setup_only=True)
    
    print_section_header("Runtime analysis results for primitives")
    for primitive in args.primitives:
        compute_execution_metrics(primitive)
        perf_utils.print_runtime_results(primitive, script_globals.primitive_perf_results[primitive])

def execute_runtime_profiling(primitive: str, event_string: str, args: argparse.Namespace, setup_only: bool) -> None:
    """Runs perf tool and processes metrics for CKKS primitives."""
    run_perf(primitive, setup_only, event_string, args)
    phase = "setup" if setup_only else "setup_and_execution"
    process_metrics_primitives(primitive, setup_only, phase)
    utils.check_metrics_primitives(primitive, phase)

        
def run_perf(
    primitive: str, setup_only: bool, event_string: str, args: argparse.Namespace
) -> None:
    """Runs the perf tool to collect performance data."""
    max_polls_per_second: int = 5000
    
    # If power and latency analysis is enabled, adjust polling frequency
    # based on the average runtime of the non-performance part
    if (args.power_latency_analysis):
        non_perf_runtime: float = (
            script_globals.setup_average_time.get(primitive, 0) if setup_only
            else script_globals.setup_and_execution_average_time.get(primitive, 0)
        )
        desired_samples: int = 10000

        polling_frequency: int = min(desired_samples / non_perf_runtime, max_polls_per_second)

        function_repeats: int = script_globals.setup_num_calls.get(primitive, 1) if setup_only \
        else script_globals.setup_and_execution_num_calls.get(primitive, 1)
    else:
        polling_frequency = max_polls_per_second
        function_repeats = 1
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
        utils.get_absolute_path(os.path.join(
            "benchmarks",
            "ckks-primitives-serialized",
            "build",
            "ckks-primitives-application",
        )),
        primitive,
        str(setup_only),
        str(args.cold_caching),
        str(function_repeats),
    ]

    # try:
    result = subprocess.run(
        cmd,
        check=True,
        cwd=utils.get_project_root(),
        stdout=None if args.verbose else subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        print(result.stderr.decode())
    # except subprocess.CalledProcessError:
    #     print_error(f"Perf record failed for {primitive}")

    output_file_path = utils.get_absolute_path(os.path.join("out", "temp", "perf_report_out.txt"))

    with open(output_file_path, "w") as outfile:
        subprocess.run(
            ["perf", "report", "--stdio", "-i", utils.get_absolute_path(os.path.join("out", "temp", "perf.data"))],
            check=True,
            stdout=outfile,
            stderr=outfile,
        )
        
def process_metrics_primitives(primitive: str, setup_only: bool, phase_suffix: str) -> None:
    """Processes metrics data collected by perf."""
    for event in script_globals.perf_events:
        perf_metric_name = event
        parse_and_extract_perf_data_primitives(
            primitive, perf_metric_name, f"{event}_{phase_suffix}"
        )

def parse_and_extract_perf_data_primitives(
    primitive: str, event_type: str, array_name: str
) -> None:
    """Parses and extracts performance data from perf output."""
    perf_out: str = utils.get_absolute_path(os.path.join("out", "temp", "perf_report_out.txt"))

    with open(perf_out, "r", encoding="utf-8") as file:
        content: str = file.read()

    event_count = re.search(
        rf"{event_type}.*?# Event count \(approx.\):\s*([0-9,]+)", content, re.DOTALL
    )
    result = event_count.group(1).replace(",", "") if event_count else "FAILURE"

    setattr(script_globals, array_name, {**getattr(script_globals, array_name, {}), primitive: result})

    
def calculate_timings_and_energies(args: argparse.Namespace) -> None:
    """Calculates the execution times, energies, and power consumption for each primitive."""
    for primitive in args.primitives:
        execution_time = abs(
            (script_globals.setup_and_execution_average_time[primitive] / script_globals.setup_and_execution_num_calls[primitive])
            - (script_globals.setup_average_time[primitive] / script_globals.setup_num_calls[primitive])
        )
        energy = abs(
            (script_globals.setup_and_execution_total_energy[primitive] / script_globals.setup_and_execution_num_calls[primitive])
            - (script_globals.setup_total_energy[primitive] / script_globals.setup_num_calls[primitive])
        )
        
        script_globals.primitive_execution_times[primitive] = execution_time
        script_globals.primitive_execution_energies[primitive] = energy
        script_globals.primitive_execution_powers[primitive] = (
            (energy / execution_time) if execution_time > 0 else float("inf")
        )


def compute_execution_metrics(primitive: str) -> None:
    """Computes execution time, rate, IPC, FLOPS, and memory bandwidth metrics."""
    
    for event in script_globals.perf_events:
        try:
            avg_setup_count = (float(getattr(script_globals, f"{event}_setup").get(primitive, 0)) 
            / script_globals.setup_num_calls.get(primitive, 1))
            avg_total_count = (float(getattr(script_globals, f"{event}_setup_and_execution").get(primitive, 0)) 
            / script_globals.setup_and_execution_num_calls.get(primitive, 1))
            execution_count = int(avg_total_count - avg_setup_count)
            
            
        except (ValueError, TypeError):
            print(f"Warning: Invalid timing value for {primitive} {event}, setting execution time to NaN")
            execution_count = float('nan')
        
        if primitive not in script_globals.primitive_perf_results:
            script_globals.primitive_perf_results[primitive] = {}
            
        script_globals.primitive_perf_results[primitive][event] = execution_count if execution_count > 0 else float('nan')

    primitive_exec_time = float(script_globals.primitive_execution_times.get(primitive, 0))
    cpu_cycles = float(script_globals.primitive_perf_results[primitive].get("cpu-cycles", 0))

    script_globals.primitive_perf_results[primitive]["memory_bandwidth"] = utils.compute_memory_bandwidth(
        primitive_exec_time,
        float(script_globals.primitive_perf_results[primitive].get("cache-misses", 0))
    )
    
    script_globals.primitive_perf_results[primitive]["ipc"] = utils.compute_ipc(
        float(script_globals.primitive_perf_results[primitive].get("instructions", 0)),
        cpu_cycles
    )
    
    if utils.check_fp_ops_enabled() is True:
        script_globals.primitive_perf_results[primitive]["flops"] = utils.compute_flops(
            primitive_exec_time,
            float(script_globals.primitive_perf_results[primitive].get("fp_ops_retired_by_type.scalar_all", 0)),
            float(script_globals.primitive_perf_results[primitive].get("fp_ops_retired_by_type.vector_all", 0)),
            float(script_globals.primitive_perf_results[primitive].get("fp_ops_retired_by_type.scalar_add", 0))
        )
            
def parse_openfhe_cryptocontext(microbenchmarking_application_output: str) -> dict:
    """Parses parameter data from the provided output string."""
    script_globals.openfhe_cryptocontext_parameters = {
        key: utils.get_match_value(pattern, microbenchmarking_application_output)
        for key, pattern in {
            "openfhe_scheme": r"scheme:\s*([^;]+)",
            "openfhe_ptModulus": r"ptModulus:\s*(\d+)",
            "openfhe_digitSize": r"digitSize:\s*(\d+)",
            "openfhe_standardDeviation": r"standardDeviation:\s*([0-9.]+)",
            "openfhe_secretKeyDist": r"secretKeyDist:\s*([^;]+)",
            "openfhe_maxRelinSkDeg": r"maxRelinSkDeg:\s*(\d+)",
            "openfhe_ksTech": r"ksTech:\s*([^;]+)",
            "openfhe_scalTech": r"scalTech:\s*([^;]+)",
            "openfhe_batchSize": r"batchSize:\s*(\d+)",
            "openfhe_firstModSize": r"firstModSize:\s*(\d+)",
            "openfhe_numLargeDigits": r"numLargeDigits:\s*(\d+)",
            "openfhe_multiplicativeDepth": r"multiplicativeDepth:\s*(\d+)",
            "openfhe_scalingModSize": r"scalingModSize:\s*(\d+)",
            "openfhe_securityLevel": r"securityLevel:\s*([^;]+)",
            "openfhe_ringDim": r"ringDim:\s*(\d+)",
            "openfhe_evalAddCount": r"evalAddCount:\s*(\d+)",
            "openfhe_keySwitchCount": r"keySwitchCount:\s*(\d+)",
            "openfhe_encryptionTechnique": r"encryptionTechnique:\s*([^;]+)",
            "openfhe_multiplicationTechnique": r"multiplicationTechnique:\s*([^;]+)",
            "openfhe_PRENumHops": r"PRENumHops:\s*(\d+)",
            "openfhe_PREMode": r"PREMode:\s*([^;]+)",
            "openfhe_multipartyMode": r"multipartyMode:\s*([^;]+)",
            "openfhe_executionMode": r"executionMode:\s*([^;]+)",
            "openfhe_decryptionNoiseMode": r"decryptionNoiseMode:\s*([^;]+)",
            "openfhe_noiseEstimate": r"noiseEstimate:\s*(\d+(?:\.\d+)?)",
            "openfhe_desiredPrecision": r"desiredPrecision:\s*(\d+(?:\.\d+)?)",
            "openfhe_statisticalSecurity": r"statisticalSecurity:\s*(\d+)",
            "openfhe_numAdversarialQueries": r"numAdversarialQueries:\s*(\d+)",
            "openfhe_thresholdNumOfParties": r"thresholdNumOfParties:\s*(\d+)",
            "openfhe_interactiveBootCompressionLevel": r"interactiveBootCompressionLevel:\s*([^\n]+)"
        }.items()
    }
    
def primitive_save_csv(args: argparse.Namespace) -> None:
    suffix = f"-{args.csv_name}" if args.csv_name else ""
    csv_output_path = utils.get_absolute_path(f"out/csv/primitive-results{suffix}.csv")
    perf_utils.save_results_csv(csv_output_path, args, 
        args.primitives, 
        script_globals.primitive_execution_times, 
        script_globals.primitive_execution_energies, 
        script_globals.primitive_execution_powers,  
        script_globals.primitive_perf_results
    )
    
def primitive_generate_flamegraph(args: argparse.Namespace) -> None:
    for primitive in args.primitives:
        print_info(f"Generating FlameGraph for {primitive}...")
        primitive_app_path = utils.get_absolute_path(os.path.join(
            "benchmarks","ckks-primitives-serialized",
            "build","ckks-primitives-application"))
        
        cmd = [
            primitive_app_path,
            primitive,
            "False",
            str(args.cold_caching),
            str(1)
        ]
        
        cwd = utils.get_project_root()
        perf_utils.generate_flamegraph(args, primitive, cmd, cwd)
