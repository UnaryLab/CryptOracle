from typing import List, Dict

class ScriptGlobals:
    def __init__(self):
        self.script_dir: str = ""
        self.cpu_count: int
        
        # Build & Execution Options
        self.cmake_opts: List[str] = []
        
        # Primitive Metrics
        self.setup_and_execution_average_time: Dict[str, float] = {}
        self.setup_and_execution_num_calls: Dict[str, int] = {}
        self.setup_average_time: Dict[str, float] = {}
        self.setup_and_execution_total_energy: Dict[str, float] = {}
        self.setup_total_energy: Dict[str, float] = {}
        self.setup_num_calls: Dict[str, int] = {}
        self.primitive_execution_times: Dict[str, float] = {}
        self.primitive_execution_energies: Dict[str, float] = {}
        self.primitive_execution_powers: Dict[str, float] = {}
        self.primitive_perf_results: Dict[str, Dict[str, float]] = {}
        
        # Microbenchmarking Metrics
        self.microbenchmarks: List[str] = []
        self.microbenchmark_setup_and_execution_times: Dict[str, float] = {}
        self.microbenchmark_setup_and_execution_energies: Dict[str, float] = {}
        self.microbenchmark_setup_and_execution_powers: Dict[str, float] = {}
        self.microbenchmark_setup_times: Dict[str, float] = {}
        self.microbenchmark_setup_energies: Dict[str, float] = {}
        self.microbenchmark_setup_powers: Dict[str, float] = {}
        self.microbenchmark_execution_times: Dict[str, float] = {}
        self.microbenchmark_execution_energies: Dict[str, float] = {}
        self.microbenchmark_execution_powers: Dict[str, float] = {}
        self.microbenchmark_setup_and_execution_perf_results: Dict[str, Dict[str, float]] = {}
        self.microbenchmark_setup_perf_results: Dict[str, Dict[str, float]] = {}
        self.microbenchmark_perf_results: Dict[str, Dict[str, float]] = {}
        self.microbenchmark_command_dict: Dict[str, str] = {}
        
        # Workload Metrics
        self.workloads: List[str] = []
        self.workload_setup_and_execution_times: Dict[str, float] = {}
        self.workload_setup_and_execution_energies: Dict[str, float] = {}
        self.workload_setup_and_execution_powers: Dict[str, float] = {}
        self.workload_setup_times: Dict[str, float] = {}
        self.workload_setup_energies: Dict[str, float] = {}
        self.workload_setup_powers: Dict[str, float] = {}
        self.workload_execution_times: Dict[str, float] = {}
        self.workload_execution_energies: Dict[str, float] = {}
        self.workload_execution_powers: Dict[str, float] = {}
        self.workload_setup_and_execution_perf_results: Dict[str, Dict[str, float]] = {}
        self.workload_setup_perf_results: Dict[str, Dict[str, float]] = {}
        self.workload_perf_results: Dict[str, Dict[str, float]] = {}
        self.workload_command_dict: Dict[str, str] = {}
        
        # System Information
        self.cpu_manufacturer: str = ""
        self.microbenchmarking_application_output: str = ""
        self.input_run_parameters = {}
        self.hardware_stats = {}
        self.openfhe_cryptocontext_parameters: Dict[str, str] = {}

        # Performance Events
        self.perf_events: List[str] = []
        
script_globals = ScriptGlobals()
