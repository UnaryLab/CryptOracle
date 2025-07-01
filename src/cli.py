import argparse
import os
import src.logging_utils as log
import sys
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
        nargs="?",
        const=True,
        type=is_boolean,
        default=False,
        help="Toggle running primitive analysis",
    )
    parser.add_argument(
        "-k",
        "--run-microbenchmarks",
        nargs="?",
        const=True,
        type=is_boolean,
        default=False,
        help="Toggle running microbenchmarks",
    )
    parser.add_argument(
        "-w",
        "--run-workloads",
        nargs="?",
        const=True,
        type=is_boolean,
        default=False,
        help="Toggle running workloads",
    )
    parser.add_argument(
        "-l",
        "--power-latency-analysis",
        nargs="?",
        const=True,
        type=is_boolean,
        default=True,
        help="Toggle running latency and power data",
    )
    parser.add_argument(
        "-r",
        "--runtime-analysis",
        nargs="?",
        const=True,
        type=is_boolean,
        default=True,
        help="Toggle runtime analysis",
    )
    parser.add_argument(
        "-f",
        "--flamegraph-generation",
        nargs="?",
        const=True,
        type=is_boolean,
        default=False,
        help="Toggle FlameGraph generation",
    )
    parser.add_argument(
        "--build",
        nargs="?",
        const=True,
        type=is_boolean,
        default=True,
        help="Toggle build/rebuild (including checks)",
    )
    parser.add_argument(
        "--compiler-optimizations",
        nargs="?",
        const=True,
        type=is_boolean,
        default=True,
        help="Toggle compiler optimizations",
    )
    parser.add_argument(
        "-e",
        "--cold-caching",
        nargs="?",
        const=True,
        type=is_boolean,
        default=True,
        help="Toggle cold-caching during primitive profiling",
    )
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
        nargs="?",
        const=True,
        type=is_boolean,
        default=False,
        help="Run a group of benchmarks based on the input .yaml file",
    ) 
    parser.add_argument(
        "--fhe",
        nargs="?",
        const=True,
        type=is_boolean,
        default=False,
        help="Enable FHE mode (True required for bootstrapping primitive)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        nargs="?",
        const=True,
        type=is_boolean,
        default=False,
        help="Enable verbose output"
    )

    args = parser.parse_args()

    # Store script directory in global variables
    script_globals.script_dir = os.path.dirname(os.path.abspath(__file__))

    return args

def print_args(args: argparse.Namespace):
    log.print_section_header("Program Inputs")
    log.print_argument("Security Standard Level:", str(args.security_standard_level))
    log.print_argument("N:", str(args.n))
    log.print_argument("Batch Size:", str(args.batch_size))
    log.print_argument("Depth:", str(args.depth))
    log.print_argument("Number of Threads:", str(args.num_threads))
    log.print_argument("Matrix Size:", str(args.matrix_size))
    log.print_argument("Primitive Profiling:", str(args.run_primitives))
    log.print_argument("Microbenchmark Profiling:", str(args.run_microbenchmarks))
    log.print_argument("Workload Profiling:", str(args.run_workloads))
    log.print_argument("Power and Latency Analysis:", str(args.power_latency_analysis))
    log.print_argument("Runtime Analysis:", str(args.runtime_analysis))
    log.print_argument("FlameGraph Generation:", str(args.flamegraph_generation))
    log.print_argument("Build:", str(args.build))
    log.print_argument("Compiler Optimizations:", str(args.compiler_optimizations))
    log.print_argument("Cold Caching:", str(args.cold_caching))
    log.print_argument("CSV Output Name:", str(args.csv_name))
    log.print_argument("Run Group:", str(args.run_group))
    log.print_argument("FHE:", str(args.fhe))
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


def is_boolean(b: str) -> bool:
    if b.lower() in ["true", "1", "yes"]:
        return True
    elif b.lower() in ["false", "0", "no"]:
        return False
    else:
        log.print_error(f"Boolean flags must be either True or False")
        sys.exit(1)