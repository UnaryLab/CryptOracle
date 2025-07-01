import argparse
import subprocess
import os
import re
import csv
import math
from datetime import datetime
from typing import Dict, List
import src.utils as utils
from src.logging_utils import print_info, print_error, print_status, print_section_header
from src.globals import script_globals


def start_perf(output_path: str) -> subprocess.Popen:
    """Starts the perf stat process to monitor power usage."""
    with open(output_path, "w") as output_file:
        return subprocess.Popen(
            ["perf", "stat", "-a", "-e", "power/energy-pkg/", "--interval-print", "5"],
            stdout=output_file,
            stderr=output_file,
        )

def stop_perf(proc: subprocess.Popen) -> None:
    """Stops the perf process."""
    if proc.poll() is None:
        proc.kill()
        proc.wait()
        
def process_metrics(target: str, perf_results: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Processes metrics data collected by perf."""
    for event in script_globals.perf_events:
        perf_metric_name = event
        parse_and_extract_perf_data(target, perf_metric_name, perf_results)
        
def parse_and_extract_perf_data(target: str, event_type: str, perf_results: Dict[str, Dict[str, float]]) -> None:
    """Parses and extracts performance data from perf output and stores in nested dictionary."""
    perf_out = utils.get_absolute_path(os.path.join("out", "temp", "perf_report_out.txt"))

    with open(perf_out, "r", encoding="utf-8") as file:
        content = file.read()

    event_count = re.search(
        rf"{event_type}.*?# Event count \(approx.\):\s*([0-9,]+)", content, re.DOTALL
    )

    result = event_count.group(1).replace(",", "") if event_count else "FAILURE"

    if target not in perf_results:
        perf_results[target] = {}

    perf_results[target][event_type] = float(result) if result != "FAILURE" else float('nan')
    
def compute_execution_metrics(microbenchmark: str, execution_time: float, setup_perf_results: Dict[str, float], setup_and_execution_perf_results: Dict[str, float], perf_results: Dict[str, float]) -> None:
    """Computes execution time, rate, and IPC"""
    for event in script_globals.perf_events:
        try:
            setup_count = float(setup_perf_results[microbenchmark].get(event, 0))
            total_count = float(setup_and_execution_perf_results[microbenchmark].get(event, 0))
            
            execution_count = total_count - setup_count
            
            perf_results[event] = execution_count if execution_count > 0 else float('nan')
            
        except (ValueError, TypeError) as e:
            print(f"Warning: Invalid value for {microbenchmark} {event}: {str(e)}")
            perf_results[event] = float('nan')
    
    cpu_cycles = float(perf_results.get("cpu-cycles", 0))

    # Calculate IPC
    perf_results["ipc"] = utils.compute_ipc(
        float(perf_results.get("instructions", 0)),
        cpu_cycles
    )
        
def print_runtime_results(target: str, perf_results: Dict[str, float]) -> None:
    """Prints the event profiling results for a single target."""
    print_info(f"{target}:")
    for event in perf_results:
        event_count = perf_results.get(event, 0)
        print_info(f"   Estimated {event}: {event_count}")

def initialize_csv_file(csv_output_file: str, target_items: List[str], args: argparse.Namespace) -> None:
    """Initializes the CSV file with headers."""
    headers = ["Date and Time of Benchmark"] + list(script_globals.hardware_stats.keys())

    headers += [
        "security_standard_level", "n", "batch_size", "depth", "num_threads",
        "runtime_analysis", "event_profiling", "flamegraph_generation", "build", "compiler_optimizations", "cold_caching"
    ]

    for target in target_items:
        headers.extend([f"{target} Time", f"{target} Energy", f"{target} Power"])
        if args.event_profiling is True:
            headers.extend([f"{target} IPC"]
            + [
                metric for event in script_globals.perf_events
                for metric in (f"{target} {event}",)
            ])

    with open(csv_output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        
def save_results_csv(csv_output_path: str,args: argparse.Namespace, target_items: List[str], 
    execution_times: Dict[str, float], 
    execution_energies: Dict[str, float], 
    execution_powers: Dict[str, float], 
    perf_results: Dict[str, Dict[str, float]]) -> None:
    """Saves benchmark results to a CSV file."""
    
    if not os.path.isfile(csv_output_path):
        initialize_csv_file(csv_output_path, target_items, args)
        
    datetime_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    csv_row = [datetime_now]
    
    csv_row += [script_globals.hardware_stats.get(key, "N/A") for key in script_globals.hardware_stats]

    # Matrix size should only be included in the csv in certain cases
    csv_row += [
        args.security_standard_level, args.n, args.batch_size, args.depth, args.num_threads, #args.matrix_size,
        args.runtime_analysis, args.event_profiling, args.flamegraph_generation, args.build, args.compiler_optimizations, args.cold_caching
    ]
    
    for target in target_items:
        csv_row.append(execution_times.get(target, "N/A"))
        csv_row.append(execution_energies.get(target, "N/A"))
        csv_row.append(execution_powers.get(target, "N/A"))
        
        if (args.event_profiling is True):
            csv_row.append(perf_results[target].get("ipc", "N/A"))
            
            for event in script_globals.perf_events:
                count = perf_results[target].get(event, "N/A")
                if isinstance(count, (int, float)):
                    csv_row.append("NaN" if math.isnan(count) else count)
                else:
                    csv_row.append("N/A")
                
    with open(csv_output_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(csv_row)

def generate_flamegraph(args: argparse.Namespace, target: str, cmd: str, cwd: str) -> None:
    """Generates a flamegraph for each CKKS operation."""
    output_dir: str = utils.get_absolute_path(os.path.join("out", "flamegraphs"))
    os.makedirs(output_dir, exist_ok=True)

    polling_frequency: int = 1000

    try:
        subprocess.run(
            [
                "perf",
                "record",
                "-o",
                utils.get_absolute_path(os.path.join("out", "temp", "perf.data")),
                "--call-graph",
                "dwarf",
                "-F",
                str(polling_frequency),
                "--",
            ] + cmd,
            cwd=cwd,
            check=True,
            stdout=None if args.verbose else subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

        subprocess.run(
            [
                "perf",
                "script",
                "-i",
                utils.get_absolute_path(os.path.join("out", "temp", "perf.data")),
            ],
            check=True,
            stdout=open(
                utils.get_absolute_path(os.path.join("out", "temp", "perf_out_flamegraph.perf")),
                "w",
            ),
            stderr=subprocess.PIPE,
        )

        subprocess.run(
            [
                utils.get_absolute_path(os.path.join("util", "flamegraph", "stackcollapse-perf.pl")),
                utils.get_absolute_path(os.path.join("out", "temp", "perf_out_flamegraph.perf")),
            ],
            check=True,
            stdout=open(
                utils.get_absolute_path(os.path.join("out", "temp", "stackcollapse_out.folded")),
                "w",
            ),
            stderr=subprocess.PIPE,
        )

        subprocess.run(
            [
                utils.get_absolute_path(os.path.join("util", "flamegraph", "flamegraph.pl")),
                "--title",
                f"{args.csv_name}-{target} FlameGraph",
                utils.get_absolute_path(os.path.join("out", "temp", "stackcollapse_out.folded")),
            ],
            check=True,
            stdout=open(
                os.path.join(output_dir, f"{args.csv_name}_{target}_FlameGraph.svg"), "w"
            ),
            stderr=subprocess.PIPE,
        )

    except subprocess.CalledProcessError as e:
        print_error(f"FlameGraph generation failed: {e.stderr.decode()}")