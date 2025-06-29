import os
import shutil
import subprocess
import argparse
import multiprocessing
import re
from typing import Dict
import src.logging_utils as log
from src.globals import script_globals


def get_project_root() -> str:
    """Returns the absolute path to the project root directory."""
    return os.path.abspath(os.path.join(script_globals.script_dir, ".."))


def get_absolute_directory(relative_path: str) -> str:
    """Returns the absolute directory path relative to the project root."""
    return os.path.join(get_project_root(), relative_path)


def command_exists(cmd: str) -> bool:
    """Checks if a command exists on the system."""
    return shutil.which(cmd) is not None


def package_installed(pkg: str) -> bool:
    """Checks if a package is installed on the system."""
    return subprocess.call(
        f"dpkg -s {pkg}", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ) == 0


def configure_multithreading(args: argparse.Namespace) -> bool:
    """Configures multithreading based on the number of threads specified."""
    script_globals.cpu_count = multiprocessing.cpu_count()
    
    if args.num_threads == 0:
        args.num_threads = multiprocessing.cpu_count()

    os.environ["OMP_NUM_THREADS"] = str(args.num_threads)
    return True


def delete_old_files() -> None:
    """Deletes old files in the temporary output directory."""
    temp_dir = get_absolute_path("out/temp")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)
    

def get_absolute_path(relative_path: str) -> str:
    """Returns the absolute path relative to the script's root directory."""
    return os.path.join(get_project_root(), relative_path)


def get_match_value(pattern: str, text: str, default: str = "") -> str:
    """Extracts a matched value using a regex pattern."""
    match = re.search(pattern, text)
    return match.group(1).strip() if match else default


def extract_float_value(output: str, key: str) -> float:
    """Extracts a float value from the output string based on a key."""
    match = re.search(rf"{key}:\s*([0-9.]+)", output)
    return float(match.group(1)) if match else float("nan")


def extract_int_value(output: str, key: str) -> int:
    """Extracts an integer value from the output string based on a key."""
    match = re.search(rf"{key}:\s*([0-9,]+)", output)
    return int(match.group(1).replace(",", "")) if match else 0


def read_file(file_path: str) -> str:
    """Reads the content of a file and returns it as a string."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        log.print_error(f"File not found: {file_path}")
        return ""
    except Exception as e:
        log.print_error(f"Error reading {file_path}: {str(e)}")
        return ""


def check_cpu_stats() -> None:
    """Collects and prints CPU and memory statistics using system commands."""
    cpu_info = {
        "CPU Vendor": "",
        "CPU Model Name": "",
        "CPU Architecture": "",
        "CPU Cores": "",
        "CPU Max MHz": "",
        "CPU Min MHz": "",
        "L1d Cache Size": "",
        "L1i Cache Size": "",
        "L2 Cache Size": "",
        "L3 Cache Size": "",
        "Total DRAM Size": "",
    }

    try:
        lscpu_output = subprocess.run(["lscpu"], capture_output=True, text=True).stdout.split("\n")

        for line in lscpu_output:
            if "Vendor ID:" in line:
                cpu_info["CPU Vendor"] = line.split(":")[1].strip()
            elif "Model name:" in line:
                cpu_info["CPU Model Name"] = line.split(":")[1].strip()
            elif "Architecture:" in line:
                cpu_info["CPU Architecture"] = line.split(":")[1].strip()
            elif "CPU(s):" in line and "NUMA" not in line:
                cpu_info["CPU Cores"] = line.split(":")[1].strip()
            elif "CPU max MHz:" in line:
                cpu_info["CPU Max MHz"] = line.split(":")[1].strip()
            elif "CPU min MHz:" in line:
                cpu_info["CPU Min MHz"] = line.split(":")[1].strip()
            elif "L1d" in line:
                cpu_info["L1d Cache Size"] = line.split(":")[1].strip()
            elif "L1i" in line:
                cpu_info["L1i Cache Size"] = line.split(":")[1].strip()
            elif "L2" in line:
                cpu_info["L2 Cache Size"] = line.split(":")[1].strip()
            elif "L3" in line:
                cpu_info["L3 Cache Size"] = line.split(":")[1].strip()
    except Exception as e:
        log.print_error(f"Error retrieving CPU information: {str(e)}")

    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if "MemTotal:" in line:
                    cpu_info["Total DRAM Size"] = str(int(line.split()[1]) // 1024) + " MB"
                    break
    except Exception as e:
        log.print_error(f"Error retrieving memory info: {str(e)}")

    script_globals.hardware_stats = cpu_info

    log.print_section_header("Processor Stats")
    for key, value in cpu_info.items():
        log.print_argument(key, value)
    print()

def check_fp_ops_enabled() -> bool:
    """Checks if floating-point operations are enabled in perf_events.yaml"""
    if ("fp_ops_retired_by_type.scalar_all" in script_globals.perf_events and "fp_ops_retired_by_type.vector_all" in script_globals.perf_events
        and "fp_ops_retired_by_type.scalar_add" in script_globals.perf_events):
        return True
    
    return False

def compute_memory_bandwidth(execution_time: float, cache_misses: int) -> float:
    """Estimates memory bandwidth based on execution time and cache misses. Result is in MB/s"""
    cache_line_size = 64
    return float(cache_line_size * cache_misses) / (1e6 * (execution_time))

def compute_ipc(instructions: int, cpu_cycles: int) -> float:
    """Computes instructions per cycle (IPC) based on instructions and CPU cycles."""
    return float(instructions / cpu_cycles)

def compute_flops(execution_time: float, scalar_ops: int, vector_ops: int, scalar_add_ops: int) -> float:
    """Computes floating-point operations per second (FLOPS) based on scalar and vector operations. Result is in MFLOPS"""
    flops = float(scalar_ops + vector_ops - scalar_add_ops) / (1e6 * (execution_time))
    return flops

def check_metrics_primitives(primitive: str, phase_suffix: str) -> None:
    """Checks and validates collected metrics."""
    for event in script_globals.perf_events:
        phase_metric = getattr(script_globals, f"{event}_{phase_suffix}")
        if phase_metric.get(primitive) in ["", "0.000", "0"]:
            phase_metric[primitive] = "FAILURE"
            
def check_metrics(target: str, perf_results: Dict[str, float]) -> None:
    for event in script_globals.perf_events:
        if perf_results.get(event) in ["", "0.000", "0", "NaN"]:
            perf_results[event] = "FAILURE"
            
def clear_cache() -> None:
    """Clears the CPU cache before running benchmarks."""
    clear_cache_cmd = get_absolute_path(os.path.join("benchmarks", "ckks-primitives", "clear-cache", "build", "clear_cache_application"))

    try:
        subprocess.run([clear_cache_cmd], check=True)
        log.print_info("Cache cleared successfully.")
    except subprocess.CalledProcessError as e:
        log.print_error(f"Failed to clear cache: {e}")