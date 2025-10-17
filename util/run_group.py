#!/usr/bin/env python3

import yaml
import subprocess
import os
from itertools import product
from typing import Dict, List, Any
from datetime import datetime
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.logging_utils import (
    print_cryptoracle_banner,
    print_section_header,
    print_status,
    print_timestamp,
    print_error,
    print_info,
    print_error
)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run microbenchmarks with different parameter combinations.')
    parser.add_argument(
        "-b",
        "--build",
        type=is_boolean,
        default=True,
        help="Toggle build/rebuild (including checks)",
    )
    parser.add_argument(
        "-n",
        "--num-runs",
        type=positive_integer, 
        default=1, 
        help="Set number of runs for each parameter combo. Default is 5."
    )
    parser.add_argument(
        "-v",
        "--verbose", 
        action="store_true", 
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "-p",
        "--run-primitives",
        type=is_boolean,
        default=False,
        help="Toggle running primitive analysis",
    )
    
    parser.add_argument(
        "-m",
        "--run-microbenchmarks",
        type=is_boolean,
        default=False,
        help="Toggle running microbenchmark analysis",
    )
    
    parser.add_argument(
        "-w",
        "--run-workloads",
        type=is_boolean,
        default=False,
        help="Toggle running workload analysis",
    )
    
    parser.add_argument(
        "-e",
        "--event-profiling",
        type=is_boolean,
        default=True,
        help="Toggle event profiling",
    )
    
    parser.add_argument(
        "-f",
        "--fhe",
        type=is_boolean,
        default=False,
        help="Toggle FHE",
    )
    
    parser.add_argument(
        "--csv-name",
        "-c",
        required=True,
        type=str,
        default="",
        help="Custom suffix for CSV output files (e.g. 'matmul' will generate 'primitive-results-matmul.csv')",
    )
    
    return parser.parse_args()

def is_boolean(b: str) -> bool:
    if b.lower() in ["true", "1", "yes"]:
        return True
    elif b.lower() in ["false", "0", "no"]:
        return False
    else:
        print_error(f"Boolean flags must be either True or False")
        sys.exit(1)
        
def positive_integer(x: str) -> int:
    value = int(x)
    if value <= 0:
        print_error(f"Value must be a positive integer")
    return value

def non_negative_integer(x: str) -> int:
    value = int(x)
    if value < 0:
        log.print_error(f"Value must be a non-negative integer")
    return value
        
def get_project_root() -> str:
    """Returns the absolute path to the project root directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_parameters(file_path: str) -> Dict[str, List[Any]]:
    """Load parameters from YAML file."""
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def run_benchmark(args: argparse.Namespace, params: Dict[str, Any], project_root: str, should_build: bool = False, should_gen_context: bool = True) -> None:
    """Run benchmark-main.py with the given parameters."""
    os.chdir(project_root)
    workloads_only = is_workloads_only(args)
    
    cmd = [
        "python3", "benchmark-main.py",
        "--build", str(should_build),
        "--num-threads", str(params['num_threads']),
        "--run-group", "True",
        "-n", str(params['n']) if not workloads_only else "1",  # n is not used for workloads, set to 1
        "--depth", str(params['depth']) if not workloads_only else "1",  # depth is not used for workloads, set to 1
        "--batch-size", str(params['batch_size']) if not workloads_only else "1",  # batch_size is not used for workloads, set to 1
        "--security-standard-level", params['security_standard_level'] if not workloads_only else "none",  # security_standard_level is not used for workloads, set to 'none'
        "--run-primitives", str(args.run_primitives),
        "--run-microbenchmarks", str(args.run_microbenchmarks),
        "--run-workloads", str(args.run_workloads), # Workloads do not use context generator, only need to run each workload on first parameter combination
        "--csv-name", str(args.csv_name),
        "--event-profiling", str(args.event_profiling),
        "--fhe", str(args.fhe)
    ]

    if args.verbose:
        cmd.extend(["--verbose"])
    
    print_section_header("Benchmark Command\n")
    
    cmd_str = ' '.join(cmd)
    cmd_len = len(cmd_str)
    split_point = cmd_len // 2
    
    while split_point < cmd_len and cmd_str[split_point] != ' ':
        split_point += 1
    
    if split_point >= cmd_len:
        split_point = cmd_len // 2
    
    print_info(f"Running: {cmd_str[:split_point]}\n         {cmd_str[split_point+1:]}")
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print_error(f"Error running benchmark: {e}")
        return

def check_parameters_changed(prev_params: Dict[str, Any], curr_params: Dict[str, Any]) -> bool:
    """Check if any of the key parameters have changed.
    
    Args:
        prev_params: Previous parameter combination
        curr_params: Current parameter combination
        
    Returns:
        bool: True if any key parameters have changed, False otherwise
    """
    key_params = ['n', 'security_standard_level', 'batch_size', 'depth']
    return any(prev_params.get(param) != curr_params.get(param) for param in key_params)

def generate_parameter_combinations(all_parameters: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    # Extract parameters to be combined
    param_names = ['n', 'security_standard_level', 'batch_size', 'depth', 'num_threads']#, 'matrix_size']
    param_values = [all_parameters.get(name, []) for name in param_names]
    
    # Generate all possible combinations
    combinations = []
    prev_params = None
    
    for combo in product(*param_values):
        params = dict(zip(param_names, combo))
        
        if params['security_standard_level'] != 'none':
            params['n'] = 12  # Default value when security standard is used. Not used by profiling script
        
        # Skip combinations where batch_size >= n
        if (params['batch_size'] >= params['n']) and params['security_standard_level'] == 'none':
            continue
        
        # 128c and 192c are equivalent to 2^16 ring dim, skip batch size > 15
        if (params['security_standard_level'] == '128c' and params['batch_size'] > 15):
            continue
        
        # 256c is equivalent to 2^17 ring dim, skip batch size > 16
        if (params['security_standard_level'] in ['192c','256c'] and params['batch_size'] > 16):
            continue
        
        # Set gen_context flag based on parameter changes
        params['gen_context'] = True if prev_params is None else check_parameters_changed(prev_params, params)
        prev_params = params.copy()
        
        combinations.append(params)
    
    return combinations

def generate_parameter_combinations_workloads(all_parameters: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    param_names = ['num_threads']
    param_values = [all_parameters.get(name, []) for name in param_names]
    
    # Generate all possible combinations
    combinations = []
    for combo in product(*param_values):
        params = dict(zip(param_names, combo))
        # Set gen_context to False for workloads
        params['gen_context'] = False
        combinations.append(params)
        
    return combinations

def is_workloads_only(args: argparse.Namespace):
    return True if str(args.run_workloads).lower() == "true" and str(args.run_microbenchmarks).lower() != "true" and str(args.run_primitives).lower() != "true" else False

def main():
    args = parse_args()
    print_cryptoracle_banner()
    
    project_root = get_project_root()
    
    if str(args.run_workloads).lower() != "true" and str(args.run_microbenchmarks).lower() != "true" and str(args.run_primitives).lower() != "true":
        print_error("At least one of --run-workloads, --run-microbenchmarks, or --run-primitives must be set to True")
        return 1
    
    params_file = os.path.join(project_root, "in", "input_parameters.yaml")
    if not os.path.exists(params_file):
        print_error(f"YAML file not found at {params_file}")
        return 1
        
    print_status(f"Loading parameter combinations from {params_file}...")
    all_parameters = load_parameters(params_file)
    
    combinations = generate_parameter_combinations_workloads(all_parameters) if is_workloads_only(args) else generate_parameter_combinations(all_parameters)
    
    print_timestamp(f"Generated {len(combinations)} parameter combinations")
    for i, combo in enumerate(combinations):
        print_info(combo)
        
    print_timestamp(f"Each combination will be run {args.num_runs} times")
    
    start_time = datetime.now()
    
    for i, values in enumerate(combinations):
        # Workloads are not parameterized, exit after each workload is run num_runs times
        # if i > 0 and is_workloads_only(args):
        #     break
        print_section_header(f"Combination {i+1}/{len(combinations)}")
        
        for run in range(args.num_runs):
            print_status(f"Run {run+1}/{args.num_runs}")
            
            should_build = args.build and i == 0 and run == 0
            if should_build:
                print_status("First run with build flag: Build parameter set to True")
            should_gen_context = True if i == 0 else False
            
            run_start = datetime.now()
            run_benchmark(args,values, project_root, should_build, should_gen_context)
            run_end = datetime.now()
            
            run_duration = (run_end - run_start).total_seconds()
            print_timestamp(f"Run {run+1} completed in {run_duration:.2f} seconds")
        
        print_timestamp(f"All runs for combination {i+1} completed")
    
    end_time = datetime.now()
    total_duration = (end_time - start_time).total_seconds()
    
    print_section_header("Summary\n")
    print_timestamp(f"All {len(combinations)} combinations completed")
    print_timestamp(f"Each combination was run {args.num_runs} times")
    print_timestamp(f"Total execution time: {total_duration:.2f} seconds")
    return 0

if __name__ == "__main__":
    main() 