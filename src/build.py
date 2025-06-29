import os
import argparse
import subprocess
import src.logging_utils as log
from src.globals import script_globals
import src.utils as utils

def build_and_install_openfhe(args: argparse.Namespace) -> None:
    """Handles the building and installation process for OpenFHE."""
    initialize_cmake_opts(args)
    append_conditional_opts(args)

    openfhe_dir = utils.get_absolute_path("openfhe-development")
    if not os.path.exists(openfhe_dir):
        log.print_error(f"OpenFHE development directory not found at: {openfhe_dir}")
        return
        
    build_dir = os.path.join(openfhe_dir, "build")
    os.makedirs(build_dir, exist_ok=True)
    
    run_cmake_and_make(args, build_dir)

    try:
        os.chdir(script_globals.script_dir)
    except OSError as e:
        log.print_error(f"Failed to re-enter root directory: {str(e)}")
        
def build_clear_cache_program(args: argparse.Namespace) -> None:
    """Handles the building and installation process for the clear cache program."""
    clear_cache_dir = utils.get_absolute_path(os.path.join("benchmarks", "ckks-primitives", "clear-cache"))
    if not os.path.exists(clear_cache_dir):
        log.print_error(f"Clear cache program directory not found at: {clear_cache_dir}")
        return
        
    build_dir = os.path.join(clear_cache_dir, "build")
    os.makedirs(build_dir, exist_ok=True)
    
    result = subprocess.run(
        ["make", f"-j{script_globals.cpu_count}"],
        cwd=build_dir,
        stdout=None if args.verbose else subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        print(result.stderr.decode())
        log.print_error("Make for clear cache program failed")


def initialize_cmake_opts(args: argparse.Namespace) -> None:
    """Initialize base CMake options."""
    script_globals.cmake_opts = [ 
        "-DBUILD_STATIC=OFF",
        "-DBUILD_SHARED=ON",
        "-DWITH_BE4=ON",
        f"-DCMAKE_INSTALL_PREFIX={utils.get_absolute_path('openfhe-development-install')}",
    ]


def append_conditional_opts(args: argparse.Namespace) -> None:
    """Append conditional options to CMake command."""
    if args.compiler_optimizations:
        script_globals.cmake_opts.append("-DWITH_NATIVEOPT=ON")
    else:
        script_globals.cmake_opts.append("-DCMAKE_BUILD_TYPE=Debug")

    if utils.configure_multithreading(args):
        script_globals.cmake_opts.extend(["-DWITH_OPENMP=ON", "-DWITH_TCM=ON"])
    else:
        script_globals.cmake_opts.extend(["-DWITH_OPENMP=OFF", "-DWITH_TCM=OFF"])
        
def run_cmake_and_make(args: argparse.Namespace, build_dir: str) -> None:
    """Run CMake and Make commands."""
    log.print_info("Running CMake for OpenFHE...")
    result = subprocess.run(
        ["cmake", *script_globals.cmake_opts, ".."], 
        cwd=build_dir,
        stdout=None if args.verbose else subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    print() if args.verbose else None
    if result.returncode != 0:
        print(result.stderr.decode())
        log.print_error("CMake configuration failed")

    if utils.configure_multithreading(args):
        log.print_info("Running Make for TCM (can take a while)...")
        result = subprocess.run(
            ["make", f"-j{script_globals.cpu_count}", "tcm"],
            cwd=build_dir,
            stdout=None if args.verbose else subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        print() if args.verbose else None
        if result.returncode != 0:
            print(result.stderr.decode())
            log.print_error("Make TCM failed")

    log.print_info("Running Make for OpenFHE (can take a while)...")
    result = subprocess.run(
        ["make", f"-j{script_globals.cpu_count}", "install"],
        cwd=build_dir,
        stdout=None if args.verbose else subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    print() if args.verbose else None
    if result.returncode != 0:
        print(result.stderr.decode())
        log.print_error("Make install failed")


def generate_serialized_files(args: argparse.Namespace) -> None:
    """Generates the serialized files for the microbenchmarks."""
    log.print_info("Building and running the cryptocontext-generator application (will take a while if FHE is enabled)...")
    project_root = utils.get_project_root()
    cryptocontext_dir = utils.get_absolute_directory(os.path.join("util", "cryptocontext-generator"))
    build_dir = os.path.join(cryptocontext_dir, "build")
    serialized_files_dir = os.path.join(cryptocontext_dir, "serialized-files")

    os.makedirs(serialized_files_dir, exist_ok=True)
    
    os.makedirs(build_dir, exist_ok=True)
    
    openfhe_install_dir = os.path.join(project_root, "openfhe-development-install")
    cmake_cmd = [
        "cmake",
        f"-DCMAKE_PREFIX_PATH={openfhe_install_dir}",
        ".."
    ]
    try:
        subprocess.run(cmake_cmd, 
                      cwd=build_dir, 
                      check=True,
                      stdout=None if args.verbose else subprocess.DEVNULL, 
                      stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        log.print_error(f"Error configuring build with CMake: {e}")
        return False

    build_cmd = ["make", f"-j{script_globals.cpu_count}"]
    try:
        subprocess.run(build_cmd, 
                      cwd=build_dir, 
                      check=True,
                      stdout=None if args.verbose else subprocess.DEVNULL, 
                      stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        log.print_error(f"Error building the application: {e}")
        return False
    
    app_cmd = ["./build/file_generator", str(args.security_standard_level), str(args.n), str(args.batch_size), str(args.depth), str(args.fhe)]
    result = subprocess.run(app_cmd, 
                    cwd=cryptocontext_dir, 
                    check=True,
                    stdout=None if args.verbose else subprocess.DEVNULL,
                    stderr=subprocess.PIPE)
    if result.returncode != 0:
        print(result.stderr.decode())
        log.print_error("Error running the application")
        return False
    
    log.print_info(f"Success! Files generated in: {serialized_files_dir}")
    
    return True