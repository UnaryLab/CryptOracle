# Dependencies:
#   System dependencies:
#       - git: For cloning and managing repositories.
#       - cmake: For building the software.
#       - autoconf: Essential for generating configuration scripts.
#       - build-essential: Installs the compiler and related tools.
#       - libtool: Necessary for compiling, linking, and managing shared libraries.
#       - libgoogle-perftools-dev: Provides the tcmalloc library for performance improvements.
#       - linux-tools-$(uname -r): For performance analysis, specific to the running kernel version.
#       - linux-tools-generic: Provides a generic Linux tools package.
#       - python3-dev: Includes development headers needed for building Python modules.
#       - python3-pip: For installing and managing Python packages.
#       - sysctl configuration change: Required to adjust security settings for performance monitoring.
#   Install commands (System):
#       sudo apt-get update -y
#       sudo apt-get upgrade -y
#       sudo apt install -y git cmake autoconf build-essential libtool libgoogle-perftools-dev python3-dev python3-pip
#       sudo apt-get install -y linux-tools-$(uname -r) linux-tools-generic
#       # Adjust /proc/sys/kernel/perf_event_paranoid setting for performance monitoring
#       echo -1 | sudo tee /proc/sys/kernel/perf_event_paranoid && echo "kernel.perf_event_paranoid = -1" | sudo tee -a /etc/sysctl.conf && sudo sysctl -p
#   Python dependencies:
#       - Python 3.x
#       - Pandas: For data manipulation and analysis
#       - Numpy: For numerical computations
#       - Matplotlib: For generating plots
#       - CSV: For writing to CSV files
#       - Random: For shuffling training data
#   Install commands (Python):
#       pip3 install pandas numpy matplotlib
#
# Further OpenFHE Documentation: https://openfhe-development.readthedocs.io/en/latest/index.html
#
# TODO: ANON REMOVAL

import argparse
from datetime import datetime
from itertools import product
from src.cli import process_cli_arguments, print_args
from src.environment import setup_env, import_run_parameters
from src.logging_utils import (
    print_status,
    print_cryptoracle_banner,
    print_timestamp,
    print_error,
    print_info,
    print_section_header,
)
from src.utils import check_cpu_stats
from src.build import generate_serialized_files
from src.primitive_profiling import analyze_primitive_performance, setup_and_build_primitive_app
from src.workload_profiling import (
    analyze_workload_performance,
    initialize_git_submodules,
    unzip_weights,
    validate_submodules,
    build_low_memory_resnet20,
    build_logistic_regression,
    build_chi_square_test,
)
from src.microbenchmark_profiling import analyze_microbenchmark_performance, build_and_install_polycircuit


def setup_benchmarking_targets(args):
    """Build all benchmark targets used for profiling without running analyses."""
    print_status("Generating serialized cryptocontext files...")
    generate_serialized_files(args)

    print_status("Building primitive benchmarking application...")
    setup_and_build_primitive_app(args)

    print_status("Building microbenchmark dependencies and binaries...")
    build_and_install_polycircuit(args)

    print_status("Initializing workload repositories...")
    initialize_git_submodules(args)
    unzip_weights()
    validate_submodules()

    print_status("Building workload binaries...")
    build_low_memory_resnet20(args)
    build_logistic_regression(args)
    build_chi_square_test(args)


def is_workloads_only(args: argparse.Namespace) -> bool:
    return args.run_workloads and not args.run_microbenchmarks and not args.run_primitives


def generate_parameter_combinations(
    all_parameters: dict, workloads_only: bool
) -> list[dict]:
    if workloads_only:
        param_names = ["num_threads"]
        return [dict(zip(param_names, combo)) for combo in product(*[all_parameters.get(name, []) for name in param_names])]

    param_names = ["n", "security_standard_level", "batch_size", "depth", "num_threads"]
    param_values = [all_parameters.get(name, []) for name in param_names]
    combinations = []

    for combo in product(*param_values):
        params = dict(zip(param_names, combo))

        if params["security_standard_level"] != "none":
            params["n"] = 12

        if params["security_standard_level"] == "none" and params["batch_size"] >= params["n"]:
            continue
        if params["security_standard_level"] == "128c" and params["batch_size"] > 15:
            continue
        if params["security_standard_level"] in ["192c", "256c"] and params["batch_size"] > 16:
            continue

        combinations.append(params)

    return combinations


def execute_single_run(args: argparse.Namespace, show_banner: bool = True, show_inputs: bool = True, show_cpu_stats: bool = True):
    start_time = datetime.now()

    if show_banner and not args.run_group:
        print_cryptoracle_banner()

    if show_inputs:
        print_args(args)

    if show_cpu_stats:
        check_cpu_stats()

    print_status("Setting up environment...")
    env_start = datetime.now()
    setup_env(args)

    if args.setup_benchmarking:
        setup_benchmarking_targets(args)
        env_end = datetime.now()
        print_timestamp(f"Environment setup completed in {(env_end - env_start).total_seconds():.3f} seconds.")
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        print_timestamp(f"Setup benchmarking flow completed in {total_duration:.3f} seconds.")
        return total_duration

    if args.run_microbenchmarks or args.run_primitives:
        generate_serialized_files(args)
        
    env_end = datetime.now()
    print_timestamp(f"Environment setup completed in {(env_end - env_start).total_seconds():.3f} seconds.")
    
    if args.run_primitives:    
        print_status("Executing primitive profiler...")     
        primitive_analysis_start = datetime.now()
        analyze_primitive_performance(args)
        primitive_analysis_end = datetime.now()
        print_timestamp(f"Primitive performance analysis completed in {(primitive_analysis_end - primitive_analysis_start).total_seconds():.3f} seconds.")
    
    if args.run_microbenchmarks:
        print_status("Executing microbenchmark profiler...")
        microbenchmark_start = datetime.now()
        analyze_microbenchmark_performance(args)
        microbenchmark_end = datetime.now()
        print_timestamp(f"Microbenchmark performance analysis completed in {(microbenchmark_end - microbenchmark_start).total_seconds():.3f} seconds.")
    
    if args.run_workloads:
        print_status("Executing workload profiler...")
        workload_start = datetime.now()
        analyze_workload_performance(args)
        workload_end = datetime.now()
        print_timestamp(f"Workload performance analysis completed in {(workload_end - workload_start).total_seconds():.3f} seconds.")
        

    end_time = datetime.now()
    total_duration = (end_time - start_time).total_seconds()
    print_timestamp(f"Profiler completed execution in {total_duration:.3f} seconds.")
    return total_duration


def run_group_mode(args: argparse.Namespace) -> None:
    if not args.run_workloads and not args.run_microbenchmarks and not args.run_primitives:
        print_error("At least one of --run-workloads, --run-microbenchmarks, or --run-primitives must be specified")
        return

    print_cryptoracle_banner()
    print_args(args)
    check_cpu_stats()

    all_parameters = import_run_parameters(args)
    workloads_only = is_workloads_only(args)
    combinations = generate_parameter_combinations(all_parameters, workloads_only)

    print_timestamp(f"Generated {len(combinations)} parameter combinations")
    for combo in combinations:
        print_info(str(combo))

    print_timestamp(f"Each combination will be run {args.num_runs} times")
    group_start = datetime.now()

    for i, params in enumerate(combinations):
        print_section_header(f"Combination {i + 1}/{len(combinations)}")

        for run in range(args.num_runs):
            print_status(f"Run {run + 1}/{args.num_runs}")
            run_args = argparse.Namespace(**vars(args))

            run_args.num_threads = params["num_threads"]
            run_args.build = args.build and i == 0 and run == 0

            if workloads_only:
                run_args.ring_dimension = 1
                run_args.depth = 1
                run_args.batch_size = 1
                run_args.security_standard_level = "none"
            else:
                run_args.ring_dimension = params["n"]
                run_args.depth = params["depth"]
                run_args.batch_size = params["batch_size"]
                run_args.security_standard_level = params["security_standard_level"]

            run_start = datetime.now()
            execute_single_run(run_args, show_banner=False, show_inputs=False, show_cpu_stats=False)
            run_end = datetime.now()
            print_timestamp(f"Run {run + 1} completed in {(run_end - run_start).total_seconds():.2f} seconds")

        print_timestamp(f"All runs for combination {i + 1} completed")

    group_end = datetime.now()
    print_section_header("Summary")
    print_timestamp(f"All {len(combinations)} combinations completed")
    print_timestamp(f"Each combination was run {args.num_runs} times")
    print_timestamp(f"Total execution time: {(group_end - group_start).total_seconds():.2f} seconds")


def main():
    args = process_cli_arguments()
    if args.setup_benchmarking:
        args.build = True

    if args.run_group:
        run_group_mode(args)
    else:
        execute_single_run(args)

if __name__ == "__main__":
    main()
