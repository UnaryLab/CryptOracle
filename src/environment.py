import yaml
import os
import subprocess
import argparse
import src.utils as utils
import src.logging_utils as log
from src.build import build_and_install_openfhe
from src.globals import script_globals


def setup_env(args: argparse.Namespace) -> None:
    """Sets up the environment by running various configuration and preparation steps."""
    if args.run_group:
        script_globals.input_run_parameters = import_run_parameters()

    if args.build:
        check_working_directory()
        ensure_git_submodules(args)
        check_required_commands()
        create_output_directories()
        build_and_install_openfhe(args)
            
    utils.delete_old_files()
    
    if args.runtime_analysis:
        import_perf_events()

    utils.configure_multithreading(args)
    log.print_info(f"Using {args.num_threads} threads for OpenMP.")


def import_perf_events() -> None:
    """Imports perf events from a YAML file."""
    file_path = utils.get_absolute_path("in/perf_events.yaml")

    try:
        with open(file_path, "r") as file:
            data = yaml.safe_load(file)
            script_globals.perf_events = data.get("perf_events", [])
    except Exception as e:
        log.print_error(f"Error loading {file_path}: {str(e)}")
        raise
    
    check_perf_event_availability()


def check_perf_event_availability() -> None:
    """Checks if the required CPU performance events are supported."""
    unsupported_events = []

    # Check each event using 'perf list' and store unsupported ones
    for perf_event in script_globals.perf_events:
        # perf_event = event.replace("_", "-")
        try:
            result = subprocess.run(
                ["perf", "list", perf_event],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if "not supported" in result.stderr or perf_event not in result.stdout:
                unsupported_events.append(perf_event)
        except Exception as e:
            log.print_error(
                f"Failed to check availability for event {perf_event}: {str(e)}"
            )

    # If any events are unsupported, print error and exit
    if unsupported_events:
        unsupported_list = ", ".join(unsupported_events)
        log.print_error(
            f"The following CPU events are not supported on this system: {unsupported_list}"
        )
        
        
def check_working_directory() -> None:
    """Ensures the working directory is set to the script's directory."""
    current_dir = os.getcwd()

    if script_globals.script_dir != current_dir:
        os.chdir(script_globals.script_dir)
        log.print_info(f"Changed working directory to: {script_globals.script_dir}")


def ensure_git_submodules(args: argparse.Namespace) -> None:
    """Ensures that required git submodules are initialized and updated."""
    submodules = {
        "util/flamegraph": "https://github.com/brendangregg/FlameGraph.git",
        "openfhe-development": "https://github.com/openfheorg/openfhe-development.git",
    }
    openfhe_version = "v1.2.4"

    for submodule, url in submodules.items():
        submodule_path = utils.get_absolute_path(submodule)
        if not os.path.isdir(submodule_path):
            log.print_info(f"Initializing submodule: {submodule}")
            try:
                subprocess.run(
                    ["git", "submodule", "update", "--init", "--recursive", submodule_path],
                    check=True,
                    stdout=None if args.verbose else subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
                log.print_info(f"Submodule {submodule} initialized successfully.")
            except subprocess.CalledProcessError as e:
                log.print_info(f"Submodule not found, adding it: {submodule}")
                try:
                    # Add submodule - only openfhe-development gets specific version
                    if submodule == "openfhe-development":
                        subprocess.run(
                            ["git", "submodule", "add", "-b", openfhe_version, url, submodule_path],
                            check=True,
                            stdout=None if args.verbose else subprocess.DEVNULL,
                            stderr=subprocess.PIPE,
                        )
                    else:
                        subprocess.run(
                            ["git", "submodule", "add", url, submodule_path],
                            check=True,
                            stdout=None if args.verbose else subprocess.DEVNULL,
                            stderr=subprocess.PIPE,
                        )
                    log.print_info(f"Submodule {submodule} added and initialized successfully.")
                except subprocess.CalledProcessError as e:
                    log.print_error(f"Failed to add submodule {submodule}: {e.stderr.decode().strip()}")
        else:
            log.print_info(f"Updating existing submodule: {submodule}")
            try:
                subprocess.run(
                    ["git", "submodule", "update", "--init", "--recursive", submodule_path],
                    check=True,
                    stdout=None if args.verbose else subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
                
                # For openfhe-development, ensure it's on the correct version
                if submodule == "openfhe-development":
                    log.print_info(f"Ensuring {submodule} is on version {openfhe_version}")
                    try:
                        # Change to submodule directory and checkout the specific version
                        subprocess.run(
                            ["git", "-C", submodule_path, "checkout", openfhe_version],
                            check=True,
                            stdout=None if args.verbose else subprocess.DEVNULL,
                            stderr=subprocess.PIPE,
                        )
                        log.print_info(f"Submodule {submodule} checked out to {openfhe_version} successfully.")
                    except subprocess.CalledProcessError as e:
                        log.print_error(f"Failed to checkout {openfhe_version} for submodule {submodule}: {e.stderr.decode().strip()}")
                
                log.print_info(f"Submodule {submodule} updated successfully.")
            except subprocess.CalledProcessError as e:
                log.print_error(f"Failed to update submodule {submodule}: {e.stderr.decode().strip()}")


def check_required_commands() -> None:
    """Verifies that required commands and system dependencies are installed."""
    required_commands = [
        "git", "cmake", "make", "sed", "autoconf", "gcc", "g++", "libtoolize"
    ]

    for cmd in required_commands:
        if not utils.command_exists(cmd):
            log.print_error(f"Required tool {cmd} is missing. Please install it.")

    if not utils.package_installed("libgoogle-perftools-dev"):
        log.print_error("Required package libgoogle-perftools-dev is missing.")

    if not utils.command_exists("perf"):
        log.print_error("perf (linux-tools) is missing. Install linux-tools-generic or linux-tools-$(uname -r).")

    verify_sysctl_settings("/proc/sys/kernel/perf_event_paranoid", -1, "kernel.perf_event_paranoid")
    verify_sysctl_settings("/proc/sys/kernel/kptr_restrict", 0, "kernel.kptr_restrict")


def verify_sysctl_settings(file_path: str, required_value: int, variable_name: str) -> None:
    """Ensures system configurations are correctly set for benchmarking."""
    if os.path.isfile(file_path):
        try:
            with open(file_path, "r") as file:
                value = int(file.read().strip())
            if value != required_value:
                log.print_error(f'{variable_name} is set to "{value}", but should be "{required_value}".')
        except Exception as e:
            log.print_error(f"Failed to read {variable_name}. Ensure permissions are correct. Error: {str(e)}")
    else:
        log.print_error(f"Missing system configuration file: {file_path}")


def create_output_directories() -> None:
    """Creates required output directories for the project."""
    output_paths = ["out", "out/temp", "out/flamegraphs", "workloads", "out/csv"]

    for path in output_paths:
        os.makedirs(utils.get_absolute_path(path), exist_ok=True)
        
            
def import_run_parameters() -> dict:
    """Imports run parameters from input_parameters.yaml."""
    file_path = utils.get_absolute_path("in/input_parameters.yaml")

    try:
        with open(file_path, "r") as file:
            data = yaml.safe_load(file)
            return {key: value if isinstance(value, list) else [value] for key, value in data.items()}
    except Exception as e:
        log.print_error(f"Error loading {file_path}: {str(e)}")
        raise
