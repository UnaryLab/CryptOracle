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

from datetime import datetime
from src.cli import process_cli_arguments, print_args
from src.environment import setup_env
from src.logging_utils import print_status, print_cryptoracle_banner, print_timestamp
from src.utils import check_cpu_stats
from src.build import generate_serialized_files
from src.primitive_profiling import analyze_primitive_performance
from src.workload_profiling import analyze_workload_performance
from src.microbenchmark_profiling import analyze_microbenchmark_performance

def main():
    start_time = datetime.now() 
    
    args = process_cli_arguments()
    
    if not args.run_group:
        print_cryptoracle_banner()
        
    print_args(args)
    check_cpu_stats()
        
    print_status("Setting up environment...")
    env_start = datetime.now()
    setup_env(args)

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

if __name__ == "__main__":
    main()

