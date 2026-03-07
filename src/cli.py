import argparse
import os
import src.logging_utils as log
from src.globals import script_globals

def process_cli_arguments() -> argparse.Namespace:
    """Processes and validates command-line arguments for the script."""
    parser = argparse.ArgumentParser(
        description="Automate benchmarking and performance analysis for OpenFHE development project."
    )

    parser.add_argument(
        "-s",
        "--security-standard-level",
        default="none",
        type=valid_security_levels,
        help="Sets or disables CKKS security standard usage",
    )
    parser.add_argument(
        "-n", 
        "--ring-dimension",
        type=positive_integer, 
        default=13, 
        help="Set CKKS parameter 'N'"
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=positive_integer,
        default=12,
        help="Set the ciphertext batching size. Minimum size of 12 for microbenchmarksand must be less than the ring dimension.",
    )
    parser.add_argument(
        "-d",
        "--depth",
        type=positive_integer,
        default=10,
        help="Set CKKS parameter 'depth'",
    )
    parser.add_argument(
        "-t",
        "--num-threads",
        type=non_negative_integer,
        default=0,
        help="Set number of OpenMP threads used by the benchmarks (0 for max available)",
    )
    parser.add_argument(
        "--matrix-size",
        type=positive_integer,
        default=32,
        help="Set the matrix size for matrix multiplication benchmarks (Supported options: 8, 16, 32, 64)",
    )
    parser.add_argument(
        "-p",
        "--run-primitives",
        action="store_true",
        default=False,
        help="Toggle running primitive analysis",
    )
    parser.add_argument(
        "-k",
        "--run-microbenchmarks",
        action="store_true",
        default=False,
        help="Toggle running microbenchmarks",
    )
    parser.add_argument(
        "-w",
        "--run-workloads",
        action="store_true",
        default=False,
        help="Toggle running workloads",
    )
    parser.add_argument(
        "-r",
        "--runtime-analysis",
        action="store_true",
        dest="runtime_analysis",
        help="Enable runtime analysis",
    )
    parser.add_argument(
        "--no-runtime-analysis",
        action="store_false",
        dest="runtime_analysis",
        help="Disable runtime analysis",
    )
    parser.set_defaults(runtime_analysis=True)
    parser.add_argument(
        "-e",
        "--event-profiling",
        action="store_true",
        dest="event_profiling",
        help="Enable event profiling",
    )
    parser.add_argument(
        "--no-event-profiling",
        action="store_false",
        dest="event_profiling",
        help="Disable event profiling",
    )
    parser.set_defaults(event_profiling=True)
    parser.add_argument(
        "-f",
        "--flamegraph-generation",
        action="store_true",
        default=False,
        help="Toggle FlameGraph generation",
    )
    parser.add_argument(
        "--build",
        action="store_true",
        dest="build",
        help="Enable build/rebuild (including checks)",
    )
    parser.add_argument(
        "--no-build",
        action="store_false",
        dest="build",
        help="Disable build/rebuild (including checks)",
    )
    parser.set_defaults(build=True)
    parser.add_argument(
        "--setup-benchmarking",
        action="store_true",
        default=False,
        help="Set up and build all OpenFHE, primitive, microbenchmark, and workload benchmarking targets without running profiling.",
    )
    parser.add_argument(
        "--compiler-optimizations",
        action="store_true",
        dest="compiler_optimizations",
        help="Enable compiler optimizations",
    )
    parser.add_argument(
        "--no-compiler-optimizations",
        action="store_false",
        dest="compiler_optimizations",
        help="Disable compiler optimizations",
    )
    parser.set_defaults(compiler_optimizations=True)
    parser.add_argument(
        "--cold-caching",
        action="store_true",
        dest="cold_caching",
        help="Enable cold-caching during primitive profiling",
    )
    parser.add_argument(
        "--no-cold-caching",
        action="store_false",
        dest="cold_caching",
        help="Disable cold-caching during primitive profiling",
    )
    parser.set_defaults(cold_caching=True)
    parser.add_argument(
        "-c",
        "--csv-name",
        type=str,
        default="",
        help="Custom suffix for CSV output files (e.g. 'matmul' will generate 'primitive-results-matmul.csv')",
    )
    parser.add_argument(
        "-g",
        "--run-group",
        action="store_true",
        default=False,
        help="Run a group of benchmarks based on the input .yaml file",
    ) 
    parser.add_argument(
        "--num-runs",
        type=positive_integer,
        default=1,
        help="Number of runs per parameter combination when --run-group is enabled.",
    )
    parser.add_argument(
        "--fhe",
        action="store_true",
        default=False,
        help="Enable FHE mode (required for bootstrapping primitive)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose output"
    )
    parser.add_argument(
        "--perf-path",
        type=str,
        default="perf",
        help="Path to perf binary to use for all perf operations.",
    )
    parser.add_argument(
        "--primitives-config",
        type=str,
        default="in/primitives.yaml",
        help="Path to primitives YAML config.",
    )
    parser.add_argument(
        "--microbenchmarks-config",
        type=str,
        default="in/microbenchmarks.yaml",
        help="Path to microbenchmarks YAML config.",
    )
    parser.add_argument(
        "--workloads-config",
        type=str,
        default="in/workloads.yaml",
        help="Path to workloads YAML config.",
    )
    parser.add_argument(
        "--perf-events-config",
        type=str,
        default="in/perf_events.yaml",
        help="Path to perf events YAML config.",
    )
    parser.add_argument(
        "--parameters-config",
        type=str,
        default="in/input_parameters.yaml",
        help="Path to run-group input parameters YAML config.",
    )

    args = parser.parse_args()

    # Store script directory in global variables
    script_globals.script_dir = os.path.dirname(os.path.abspath(__file__))
    script_globals.perf_path = args.perf_path

    return args

def print_args(args: argparse.Namespace):
    log.print_section_header("Program Inputs")
    log.print_argument("Security Standard Level:", str(args.security_standard_level))
    log.print_argument("N:", str(args.ring_dimension))
    log.print_argument("Batch Size:", str(args.batch_size))
    log.print_argument("Depth:", str(args.depth))
    log.print_argument("Number of Threads:", str(args.num_threads))
    log.print_argument("Matrix Size:", str(args.matrix_size))
    log.print_argument("Primitive Profiling:", str(args.run_primitives))
    log.print_argument("Microbenchmark Profiling:", str(args.run_microbenchmarks))
    log.print_argument("Workload Profiling:", str(args.run_workloads))
    log.print_argument("Runtime Analysis:", str(args.runtime_analysis))
    log.print_argument("Event Profiling:", str(args.event_profiling))
    log.print_argument("FlameGraph Generation:", str(args.flamegraph_generation))
    log.print_argument("Build:", str(args.build))
    log.print_argument("Setup Benchmarking:", str(args.setup_benchmarking))
    log.print_argument("Compiler Optimizations:", str(args.compiler_optimizations))
    log.print_argument("Cold Caching:", str(args.cold_caching))
    log.print_argument("CSV Output Name:", str(args.csv_name))
    log.print_argument("Run Group:", str(args.run_group))
    log.print_argument("Run Group Num Runs:", str(args.num_runs))
    log.print_argument("FHE:", str(args.fhe))
    log.print_argument("Perf Path:", str(args.perf_path))
    log.print_argument("Primitives Config:", str(args.primitives_config))
    log.print_argument("Microbenchmarks Config:", str(args.microbenchmarks_config))
    log.print_argument("Workloads Config:", str(args.workloads_config))
    log.print_argument("Perf Events Config:", str(args.perf_events_config))
    log.print_argument("Input Parameters Config:", str(args.parameters_config))
    log.print_argument("Verbosity:", str(args.verbose))
    print()
    
def valid_security_levels(s: str) -> str:
    if s not in ["none", "128c", "192c", "256c", "128q", "192q", "256q"]:
        log.print_error(
            "Security level must be one of: none, 128c, 192c, 256c, 128q, 192q, 256q"
        )
    return s

def positive_integer(x: str) -> int:
    value = int(x)
    if value <= 0:
        log.print_error(f"Value must be a positive integer")
    return value


def non_negative_integer(x: str) -> int:
    value = int(x)
    if value < 0:
        log.print_error(f"Value must be a non-negative integer")
    return value
