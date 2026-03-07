"""Generate ISPASS artifact-evaluation plots from bundled CSV datasets."""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = SCRIPT_DIR
CSV_DIR = REPO_ROOT / "ispass-datasets"
with_legend = False

# Primitive Data
AMD_PRIMITIVE_RUNTIME_CSV                       = f"{CSV_DIR}/amd/primitives/all-primitives-timing-results.csv"
AMD_PRIMITIVE_EVENT_CSV                         = f"{CSV_DIR}/amd/primitives/primitive-results-perf-events.csv"
AMD_PRIMITIVES_HARDWARE_BACKEND_CSV             = f"{CSV_DIR}/amd/primitives/primitive-results-ispass-hexl-data.csv"
INTEL_PRIMITIVE_RUNTIME_CSV                      = f"{CSV_DIR}/intel/primitives/primitive-results.csv"
INTEL_PRIMITIVE_EVENT_CSV                       = f"{CSV_DIR}/intel/primitives/primitive-results.csv"

# Microbenchmark Data
AMD_MICROBENCH_CSV                              = f"{CSV_DIR}/amd/microbenchmarks/microbenchmark-results.csv"
AMD_MICROBENCH_MATMUL_SIZES_CSV                 = f"{CSV_DIR}/amd/microbenchmarks/microbenchmarks-different-sized-matrices.csv"
INTEL_MICROBENCH_NO_SEC_STD_CSV                 = f"{CSV_DIR}/intel/microbenchmarks/microbenchmark-results.csv"

# Workload Data
AMD_WORKLOAD_CSV                                = f"{CSV_DIR}/amd/workloads/all-workloads-no-resnet-results.csv"
AMD_RESNET_CSV                                  = f"{CSV_DIR}/amd/workloads/resnet-threads-sweep.csv"
AMD_CHI_SQUARE_CSV                              = f"{CSV_DIR}/amd/workloads/chi-square-results.csv"

# Figures 9-10
MATMUL_ESTIMATED_RUNTIME_CSV                    = f"{CSV_DIR}/perf-model/matmul/estimated_matrix_multiplication_Time_median.csv"
LOGISTIC_FUNCTION_ESTIMATED_RUNTIME_CSV         = f"{CSV_DIR}/perf-model/logistic_function/estimated_logistic_function_Time_median.csv"
SIGN_EVAL_ESTIMATED_RUNTIME_CSV                 = f"{CSV_DIR}/perf-model/sign_eval/estimated_sign_eval_Time_median.csv"

MATMUL_ESTIMATED_ENERGY_CSV                     = f"{CSV_DIR}/perf-model/matmul/estimated_matrix_multiplication_Energy_median.csv"
LOGISTIC_FUNCTION_ESTIMATED_ENERGY_CSV          = f"{CSV_DIR}/perf-model/logistic_function/estimated_logistic_function_Energy_median.csv"
SIGN_EVAL_ESTIMATED_ENERGY_CSV                  = f"{CSV_DIR}/perf-model/sign_eval/estimated_sign_eval_Energy_median.csv"

RESNET_ESTIMATED_RUNTIME_CSV                    = f"{CSV_DIR}/perf-model/resnet/estimated_low_memory_resnet20_Time_median.csv"
LOGISTIC_REGRESSION_ESTIMATED_RUNTIME_CSV       = f"{CSV_DIR}/perf-model/logreg/estimated_logistic_regression_Time_median.csv"
CIFAR_10_ESTIMATED_RUNTIME_CSV                  = f"{CSV_DIR}/perf-model/cifar10/estimated_cifar10_Time_median.csv"
CHI_SQUARE_ESTIMATED_RUNTIME_CSV                = f"{CSV_DIR}/perf-model/chi-square/estimated_chi_square_test_Time_median.csv"

RESNET_ESTIMATED_ENERGY_CSV                     = f"{CSV_DIR}/perf-model/resnet/estimated_low_memory_resnet20_Energy_median.csv"
LOGISTIC_REGRESSION_ESTIMATED_ENERGY_CSV        = f"{CSV_DIR}/perf-model/logreg/estimated_logistic_regression_Energy_median.csv"
CIFAR_10_ESTIMATED_ENERGY_CSV                   = f"{CSV_DIR}/perf-model/cifar10/estimated_cifar10_Energy_median.csv"
CHI_SQUARE_ESTIMATED_ENERGY_CSV                 = f"{CSV_DIR}/perf-model/chi-square/estimated_chi_square_test_Energy_median.csv"

# Figure 10
MATMUL_PRIMITIVE_CONTRIBUTION_CSV               = f"{CSV_DIR}/function-breakdowns/matmul.csv"
LOGISTIC_FUNCTION_PRIMITIVE_CONTRIBUTION_CSV    = f"{CSV_DIR}/function-breakdowns/logistic_function.csv"
SIGN_EVALUATION_CONTRIBUTION_CSV                = f"{CSV_DIR}/function-breakdowns/sign_eval.csv"
RESNET_PRIMITIVE_CONTRIBUTION_CSV               = f"{CSV_DIR}/function-breakdowns/resnet.csv"
LOGISTIC_REGRESSION_PRIMITIVE_CONTRIBUTION_CSV  = f"{CSV_DIR}/function-breakdowns/logreg.csv"
CIFAR10_PRIMITIVE_CONTRIBUTION_CSV              = f"{CSV_DIR}/function-breakdowns/cifar10.csv"
CHI_SQUARE_PRIMITIVE_CONTRIBUTION_CSV           = f"{CSV_DIR}/function-breakdowns/chi_square_test.csv"

# Figure 11
AMD_TIME_ORIGINAL_ALGO_CSV                      = f"{CSV_DIR}/perf-model/figure_11/amd_estimated_runtime_matmul_original_algo_median.csv"
AMD_TIME_NEW_ALGO_CSV                           = f"{CSV_DIR}/perf-model/figure_11/amd_estimated_runtime_matmul_new_algo_median.csv"
AMD_ENERGY_ORIGINAL_ALGO_CSV                    = f"{CSV_DIR}/perf-model/figure_11/amd_estimated_energy_matmul_original_algo_median.csv"
AMD_ENERGY_NEW_ALGO_CSV                         = f"{CSV_DIR}/perf-model/figure_11/amd_estimated_energy_matmul_new_algo_median.csv"

INTEL_TIME_ORIGINAL_ALGO_CSV                    = f"{CSV_DIR}/perf-model/figure_11_intel/estimated_intel_matmul_original_algo_Time_median.csv"
INTEL_TIME_NEW_ALGO_CSV                         = f"{CSV_DIR}/perf-model/figure_11_intel/estimated_intel_matmul_new_algo_Time_median.csv"
INTEL_ENERGY_ORIGINAL_ALGO_CSV                  = f"{CSV_DIR}/perf-model/figure_11_intel/estimated_intel_matmul_original_algo_Energy_median.csv"
INTEL_ENERGY_NEW_ALGO_CSV                       = f"{CSV_DIR}/perf-model/figure_11_intel/estimated_intel_matmul_new_algo_Energy_median.csv"

estimated_vs_actual_time_dict: Dict[str, Dict[str, Dict[str, str]]] = {
    "AMD": {
        "estimated": {
            "matrix_multiplication": MATMUL_ESTIMATED_RUNTIME_CSV,
            "logistic_function": LOGISTIC_FUNCTION_ESTIMATED_RUNTIME_CSV,
            "sign_eval": SIGN_EVAL_ESTIMATED_RUNTIME_CSV,
            "chi_square_test": CHI_SQUARE_ESTIMATED_RUNTIME_CSV,
            "low_memory_resnet20": RESNET_ESTIMATED_RUNTIME_CSV,
            "logistic_regression": LOGISTIC_REGRESSION_ESTIMATED_RUNTIME_CSV,
            "cifar10": CIFAR_10_ESTIMATED_RUNTIME_CSV
        },
        "actual": {
            "matrix_multiplication": AMD_MICROBENCH_CSV,
            "logistic_function": AMD_MICROBENCH_CSV,
            "sign_eval": AMD_MICROBENCH_CSV,
            "chi_square_test": AMD_CHI_SQUARE_CSV,
            "low_memory_resnet20": AMD_RESNET_CSV,
            "logistic_regression": AMD_WORKLOAD_CSV,
            "cifar10": AMD_WORKLOAD_CSV
        }
    },
    "INTEL": {
        "estimated": {
            "matrix_multiplication": None,
            "logistic_function": None,
            "sign_eval": None,
            "chi_square_test": None,
            "low_memory_resnet20": None,
            "logistic_regression": None,
            "cifar10": None
        },
        "actual": {
            "matrix_multiplication": INTEL_MICROBENCH_NO_SEC_STD_CSV,
            "logistic_function": INTEL_MICROBENCH_NO_SEC_STD_CSV,
            "sign_eval": INTEL_MICROBENCH_NO_SEC_STD_CSV,
            "chi_square_test":  None,
            "low_memory_resnet20": None,
            "logistic_regression": None,
            "cifar10": None
        }
    }
}

estimated_vs_actual_energy_dict: Dict[str, Dict[str, Dict[str, str]]] = {
    "AMD": {
        "estimated": {
            "matrix_multiplication": MATMUL_ESTIMATED_ENERGY_CSV,
            "logistic_function": LOGISTIC_FUNCTION_ESTIMATED_ENERGY_CSV,
            "sign_eval": SIGN_EVAL_ESTIMATED_ENERGY_CSV,
            "chi_square_test": CHI_SQUARE_ESTIMATED_ENERGY_CSV,
            "low_memory_resnet20": RESNET_ESTIMATED_ENERGY_CSV,
            "logistic_regression": LOGISTIC_REGRESSION_ESTIMATED_ENERGY_CSV,
            "cifar10": CIFAR_10_ESTIMATED_ENERGY_CSV
        },
        "actual": {
            "matrix_multiplication": AMD_MICROBENCH_CSV,
            "logistic_function": AMD_MICROBENCH_CSV,
            "sign_eval": AMD_MICROBENCH_CSV,
            "chi_square_test": AMD_CHI_SQUARE_CSV,
            "low_memory_resnet20": AMD_RESNET_CSV,
            "logistic_regression": AMD_WORKLOAD_CSV,
            "cifar10": AMD_WORKLOAD_CSV
        }
    },
    "INTEL": {
        "estimated": {
            "matrix_multiplication": None,
            "logistic_function": None,
            "sign_eval": None,
            "chi_square_test": None,
            "low_memory_resnet20": None,
            "logistic_regression": None,
            "cifar10": None
        },
        "actual": {
            "matrix_multiplication": INTEL_MICROBENCH_NO_SEC_STD_CSV,
            "logistic_function": INTEL_MICROBENCH_NO_SEC_STD_CSV,
            "sign_eval": INTEL_MICROBENCH_NO_SEC_STD_CSV,
            "chi_square_test":  None,
            "low_memory_resnet20": None,
            "logistic_regression": None,
            "cifar10": None
        }
    }
}

runtime_contributions_dict: Dict[str, Dict[str, str]] = {
    "estimated": {
        "matrix_multiplication": MATMUL_ESTIMATED_RUNTIME_CSV,
        "logistic_function": LOGISTIC_FUNCTION_ESTIMATED_RUNTIME_CSV,
        "sign_eval": SIGN_EVAL_ESTIMATED_RUNTIME_CSV,
        "chi_square_test": CHI_SQUARE_ESTIMATED_RUNTIME_CSV,
        "low_memory_resnet20": RESNET_ESTIMATED_RUNTIME_CSV,
        "logistic_regression": LOGISTIC_REGRESSION_ESTIMATED_RUNTIME_CSV,
        "cifar10": CIFAR_10_ESTIMATED_RUNTIME_CSV
    },
    "actual": {
        "matrix_multiplication": MATMUL_PRIMITIVE_CONTRIBUTION_CSV,
        "logistic_function": LOGISTIC_FUNCTION_PRIMITIVE_CONTRIBUTION_CSV,
        "sign_eval": SIGN_EVALUATION_CONTRIBUTION_CSV,
        "chi_square_test": CHI_SQUARE_PRIMITIVE_CONTRIBUTION_CSV,
        "low_memory_resnet20": RESNET_PRIMITIVE_CONTRIBUTION_CSV,
        "logistic_regression": LOGISTIC_REGRESSION_PRIMITIVE_CONTRIBUTION_CSV,
        "cifar10": CIFAR10_PRIMITIVE_CONTRIBUTION_CSV
    }
}

energy_contributions_dict: Dict[str, Dict[str, str]] = {
    "estimated": {
        "matrix_multiplication": MATMUL_ESTIMATED_ENERGY_CSV,
        "logistic_function": LOGISTIC_FUNCTION_ESTIMATED_ENERGY_CSV,
        "sign_eval": SIGN_EVAL_ESTIMATED_ENERGY_CSV,
        "chi_square_test": CHI_SQUARE_ESTIMATED_ENERGY_CSV,
        "low_memory_resnet20": RESNET_ESTIMATED_ENERGY_CSV,
        "logistic_regression": LOGISTIC_REGRESSION_ESTIMATED_ENERGY_CSV,
        "cifar10": CIFAR_10_ESTIMATED_ENERGY_CSV
    },
    "actual": {
        "matrix_multiplication": MATMUL_PRIMITIVE_CONTRIBUTION_CSV,
        "logistic_function": LOGISTIC_FUNCTION_PRIMITIVE_CONTRIBUTION_CSV,
        "sign_eval": SIGN_EVALUATION_CONTRIBUTION_CSV,
        "chi_square_test": CHI_SQUARE_PRIMITIVE_CONTRIBUTION_CSV,
        "low_memory_resnet20": RESNET_PRIMITIVE_CONTRIBUTION_CSV,
        "logistic_regression": LOGISTIC_REGRESSION_PRIMITIVE_CONTRIBUTION_CSV,
        "cifar10": CIFAR10_PRIMITIVE_CONTRIBUTION_CSV
    }
}

# Composition of Primitives for each microbenchmark/workload
primitives_dict: Dict[str, List[str]] = {
    "matrix_multiplication": [
        "CKKS_Add", "CKKS_Mult", "CKKS_Mult_Plaintext", "CKKS_Rotate"
    ],
    "logistic_function": [
        "CKKS_Add", "CKKS_Sub", "CKKS_Sub_Scalar", "CKKS_Square", "CKKS_Mult", "CKKS_Mult_Scalar", "CKKS_Chebyshev_Series"
    ],
    "sign_eval": [
        "CKKS_Add", "CKKS_Sub", "CKKS_Sub_Scalar", "CKKS_Mult", "CKKS_Mult_Scalar", "CKKS_Chebyshev_Series"
    ],
    "chi_square_test": [
        "CKKS_MultNoRelin", "CKKS_Add", "CKKS_ModReduce", "CKKS_LevelReduce", "CKKS_Mult_Plaintext", "CKKS_Sub", "CKKS_Mult_Scalar"
    ],
    "low_memory_resnet20": [
        "CKKS_Add", "CKKS_Mult_Plaintext", "CKKS_Bootstrap", "CKKS_Chebyshev_Function", "CKKS_Fast_Rotate_Precompute",
        "CKKS_Rotate", "CKKS_Fast_Rotate", "CKKS_Add_Plaintext", "CKKS_Add_Many", "CKKS_Mult"
    ],
    "logistic_regression": [
        "CKKS_Mult_Plaintext", "CKKS_Add", "CKKS_Rotate", "CKKS_Bootstrap", "CKKS_Sub", "CKKS_Mult", "CKKS_Logistic"
    ],
    "cifar10": [
        "CKKS_Add", "CKKS_Add_Plaintext", "CKKS_Sub", "CKKS_Sub_Scalar", "CKKS_Mult", "CKKS_Mult_Plaintext", "CKKS_Rotate"
    ]
}

# Figure Dimension Parameters
BASE_TEXTWIDTH = 6.5 
FIG_WIDTH = BASE_TEXTWIDTH*.25
FIG_HEIGHT = BASE_TEXTWIDTH * 0.25

FONT_SIZE = 8
TICK_SIZE = 8
TITLE_SIZE = 8
SUBTITLE_SIZE = 8

primitives_label_map: Dict[str, str] = {
    "CKKS_Add": "EvalAdd",
    "CKKS_Add_Plaintext": "EvalAdd (Ptxt)",
    "CKKS_Mult": "EvalMult",
    "CKKS_Mult_Plaintext": "EvalMult (Ptxt)",
    "CKKS_Rotate": "EvalRotate",
    "CKKS_Sub": "EvalSub",
    "CKKS_Sub_Scalar": "EvalSub (Scalar)",
    "CKKS_Square": "EvalSquare",
    "CKKS_Mult_Scalar": "EvalMult (Scalar)",
    "CKKS_Chebyshev_Series": "EvalChebyshevSeries",
    "CKKS_Bootstrap": "EvalBootstrap",
    "CKKS_Fast_Rotate_Precompute": "EvalFastRotatePrecompute",
    "CKKS_Fast_Rotate": "EvalFastRotate",
    "CKKS_Add_Many": "EvalAddMany",
    "CKKS_Chebyshev_Function": "EvalChebyshevFunction",
    "CKKS_Logistic": "EvalLogistic",
    "CKKS_MultNoRelin": "EvalMultNoRelin",
    "CKKS_ModReduce": "ModReduce",
    "CKKS_LevelReduce": "LevelReduce"
}

event_label_map: Dict[str, str] = {
    "instructions": "Instructions",
    "cache-references": "Cache References",
    "dTLB-loads": "dTLB Loads",
    "page-faults": "Page Faults"
}

application_label_map: Dict[str, str] = {
    "matrix_multiplication": "matmul",
    "logistic_function": "logistic-func",
    "sign_eval": "sign-eval",
    "chi_square_test": "chi-square",
    "low_memory_resnet20": "resnet",
    "logistic_regression": "logreg",
    "cifar10": "cifar10"
}


# Figure 5: Primitive runtime and energy across different CPU vendors and thread counts (x axis). Use curves for each primitives
def figure_5_primitive_curve_thread_sweep(ax1, tag, set_legend, df, platform: str, primitives: List[str], metric: str = "Time", output_dir: str = OUTPUT_DIR) -> None:
    # fig, ax1 = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT * (2 if with_legend else 1)))
    ax2 = ax1.twinx()
    
    # Compute a threshold (mean of all y values across all primitives)
    all_y = []
    for primitive in primitives:
        grouped_df = group_by_column(df, "num_threads", primitive, metric)
        y = grouped_df[f"{primitive} {metric}"]
        if metric == "Time":
            y = y * 1000
        all_y.extend(y.values)
    threshold = np.mean(all_y)

    idx1, idx2 = 0, 0
    colors1 = plt.cm.cool(np.linspace(0, 1, len(primitives)))  # maximally distinct warm colors for cold group
    colors2 = plt.cm.autumn(np.linspace(0, .8, 2))  # maximally distinct warm colors for warm group. Somewhat arbitrary choice of 2
    
    for primitive in primitives:
        grouped_df = group_by_column(df, "num_threads", primitive, metric)
        y = grouped_df[f"{primitive} {metric}"]
        if metric == "Time":
            y = y * 1000
        x = grouped_df["num_threads"].astype(str)
        if y.max() > threshold:
            ax2.plot(x, y, label=primitives_label_map[primitive], color=colors2[idx2], linestyle='--')
            idx2 += 1
        else:
            ax1.plot(x, y, label=primitives_label_map[primitive], color=colors1[idx1])
            idx1 += 1
    
    # ax1.set_xlabel("Number of Threads")
    ax1.set_ylabel("Runtime (ms)" if metric == "Time" else "Energy (J)", fontsize=FONT_SIZE)
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax1.grid(True)
    
    # Set more major and minor y-ticks and gridlines
    ax1.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
    ax1.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax2.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
    ax2.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    
    # Legends for both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    if set_legend:
        ax1.legend(
            lines1 + lines2,
            labels1 + labels2,
            # title="Primitives",
            loc='lower center',
            bbox_to_anchor=(1.3, 1.3),
            fontsize=FONT_SIZE,
            title_fontsize=FONT_SIZE,
            ncols=3,
            handlelength=1.2,           # shorten legend line
            labelspacing=0.0,
            columnspacing=0.5
        )
    ax1.set_title(f"{tag} {platform} - {metric}", fontsize=TITLE_SIZE, pad=3)


# Figure 6: Primitive architecture stats across different CPU vendors and thread counts (x axis). Same primitives as figure 5. Use curves for each primitive
def figure_6_primitive_curve_event_stats_thread_sweep(ax1, tag, set_legend, df, platform: str, primitives: List[str], event: str = "Time", output_dir: str = OUTPUT_DIR) -> None:
    # fig, ax1 = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT * (2 if with_legend else 1)))
    ax2 = ax1.twinx()
    
    # Compute a threshold (mean of all y values across all primitives)
    all_y = []
    for primitive in primitives:
        grouped_df = group_by_column(df, "num_threads", primitive, event)
        y = grouped_df[f"{primitive} {event}"]
        all_y.extend(y.values)
    threshold = np.mean(all_y)

    idx1, idx2 = 0, 0
    colors1 = plt.cm.cool(np.linspace(0, 1, len(primitives)))  # maximally distinct warm colors for cold group
    colors2 = plt.cm.autumn(np.linspace(0, .8, 2))  # maximally distinct warm colors for warm group. Somewhat arbitrary choice of 2
    
    for primitive in primitives:
        grouped_df = group_by_column(df, "num_threads", primitive, event)
        y = grouped_df[f"{primitive} {event}"]
        x = grouped_df["num_threads"].astype(str)
        if y.max() > threshold:
            ax2.plot(x, y, label=primitives_label_map[primitive], color=colors2[idx2], linestyle='--')
            idx2 += 1
        else:
            ax1.plot(x, y, label=primitives_label_map[primitive], color=colors1[idx1])
            idx1 += 1
        
    ax1.set_ylabel(event_label_map[event], fontsize=FONT_SIZE)
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax1.grid(True)
    
    # Set more major and minor y-ticks and gridlines
    ax1.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
    ax1.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax2.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
    ax2.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    
    # Legends for both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    if set_legend:
        ax1.legend(
            lines1 + lines2,
            labels1 + labels2,
            # title="Primitives",
            loc='lower center',
            bbox_to_anchor=(1.3, 1.3),
            fontsize=FONT_SIZE,
            title_fontsize=FONT_SIZE,
            ncols=3,
            handlelength=1.2,           # shorten legend line
            labelspacing=0.0,
            columnspacing=0.5
        )
    ax1.set_title(f"{tag} {event}", fontsize=TITLE_SIZE, pad=3)


# Figure 7: Microbench runtime and energy across different CPU vendors and thread counts (x axis). Matmul size is NxN. Use curves for each microbenchmark, each a different color
def figure_7_microbenchmark_curve_thread_sweep(ax, tag, set_legend, df, platform: str, microbenchmarks: List[str], metric: str = "Time", output_dir: str = OUTPUT_DIR) -> None:
    # fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT * (2 if with_legend else 1)))
    for microbenchmark in microbenchmarks:
        grouped_df = group_by_column(df, "num_threads", microbenchmark, metric)
        y = grouped_df[f"{microbenchmark} {metric}"]
        ax.plot(grouped_df["num_threads"].astype(str), y, label=microbenchmark)
    if (metric == "Time"):
        ax.set_ylabel("Runtime (s)")
    elif (metric == "Energy"):
        ax.set_ylabel("Energy (J)")
    # ax.set_xlabel("Number of Threads")
    ax.grid(True)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    if set_legend:
        ax.legend(
            # title="Microbenchmarks",
            loc='lower center',
            bbox_to_anchor=(1.3, 1.3),
            fontsize=FONT_SIZE,
            title_fontsize=FONT_SIZE,
            ncols=2,
            handlelength=1.2,           # shorten legend line
            labelspacing=0.0,
            columnspacing=0.5
        )
    
    ax.tick_params(axis='y', labelcolor='black')
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax.set_title(f"{tag} {platform} - {metric}", fontsize=TITLE_SIZE, pad=3)

#Figure 8 (in ISPASS paper): Primitive runtime and energy across different hardware backends and ring dimension counts (x axis). Use curves for each primitives
def figure_8_primitive_hardware_backend_curve_ring_sweep(ax1, tag, set_legend, df, platform: str, primitives: List[str], metric: str = "Time", output_dir: str = OUTPUT_DIR) -> None:
    # Normalize hexl column to boolean if needed
    if "hexl" in df.columns:
        if df["hexl"].dtype == 'object':
            df = df.copy()
            df["hexl"] = df["hexl"].astype(str).str.lower().map({'true': True, 'false': False, '1': True, '0': False})

    colors_non_hexl = plt.cm.cool(np.linspace(0, 1, len(primitives)))  # colors for non-HEXL (solid lines)
    colors_hexl = plt.cm.autumn(np.linspace(0, .8, len(primitives)))  # colors for HEXL (dashed lines)
    
    for idx, primitive in enumerate(primitives):
        color_non_hexl = colors_non_hexl[idx] if idx < len(colors_non_hexl) else colors_non_hexl[0]
        color_hexl = colors_hexl[idx] if idx < len(colors_hexl) else colors_hexl[0]
        
        # Plot hexl=False (solid line, non-HEXL color)
        df_hexl_false = df[df["hexl"] == False] if "hexl" in df.columns else df
        if len(df_hexl_false) > 0:
            grouped_df_false = group_by_column(df_hexl_false, "n", primitive, metric)
            y_false = grouped_df_false[f"{primitive} {metric}"]
            if metric == "Time":
                y_false = y_false * 1000
            x_false = grouped_df_false["n"].astype(str)
            ax1.plot(x_false, y_false, label=f"{primitives_label_map[primitive]} (no HEXL)", color=color_non_hexl, linestyle='-')
        
        # Plot hexl=True (dashed line, HEXL color)
        df_hexl_true = df[df["hexl"] == True] if "hexl" in df.columns else df
        if len(df_hexl_true) > 0:
            grouped_df_true = group_by_column(df_hexl_true, "n", primitive, metric)
            y_true = grouped_df_true[f"{primitive} {metric}"]
            if metric == "Time":
                y_true = y_true * 1000
            x_true = grouped_df_true["n"].astype(str)
            ax1.plot(x_true, y_true, label=f"{primitives_label_map[primitive]} (HEXL)", color=color_hexl, linestyle='--')
    
    # Calculate speedups across all ring dimensions.
    speedups = []
    for idx, primitive in enumerate(primitives):
        df_hexl_false = df[df["hexl"] == False] if "hexl" in df.columns else df
        df_hexl_true = df[df["hexl"] == True] if "hexl" in df.columns else df
        if len(df_hexl_false) > 0 and len(df_hexl_true) > 0:
            grouped_false = group_by_column(df_hexl_false, "n", primitive, metric)
            grouped_true = group_by_column(df_hexl_true, "n", primitive, metric)
            for n_val in grouped_false["n"]:
                if n_val in grouped_true["n"].values:
                    time_false = grouped_false.loc[grouped_false["n"] == n_val, f"{primitive} {metric}"].values[0]
                    time_true = grouped_true.loc[grouped_true["n"] == n_val, f"{primitive} {metric}"].values[0]
                    if time_true > 0:
                        speedup = time_false / time_true
                        speedups.append(speedup)
    if speedups:
        _ = max(speedups)
        _ = sum(speedups) / len(speedups)
    
    # ax1.set_xlabel("Number of Threads")
    ax1.set_ylabel("Runtime (ms)" if metric == "Time" else "Energy (J)", fontsize=FONT_SIZE)
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax1.grid(True)
    
    # Set more major and minor y-ticks and gridlines
    ax1.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
    ax1.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    
    # Legend
    if set_legend:
        ax1.legend(
            loc='lower center',
            bbox_to_anchor=(1.3, 1.3),
            fontsize=FONT_SIZE,
            title_fontsize=FONT_SIZE,
            ncols=3,
            handlelength=1.2,           # shorten legend line
            labelspacing=0.0,
            columnspacing=0.5
        )
    ax1.set_title(f"{tag}", fontsize=TITLE_SIZE, pad=3)
    
#Figure 9: Matrix multiplication runtime on AMD CPU, Gemm sizes, ring size (x axis) and thread counts (curves)
def figure_9_matmul_thread_curve_ring_sweep(ax, tag, set_legend, df, platform: str, matrix_size: int, ring_dimensions: List[int], num_threads: List[int], metric: str = "Time", output_dir: str = OUTPUT_DIR) -> None:
    # fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT * (2 if with_legend else 1)))
    filtered_df = df[df["matrix_size"] == matrix_size]
    for thread in num_threads:
        thread_df = filtered_df[filtered_df["num_threads"] == thread]
        grouped_df = group_by_column(thread_df, "n", "matrix_multiplication", metric)
        y = grouped_df[f"matrix_multiplication {metric}"]
        ax.plot(grouped_df["n"].astype(str), y, label=f"{thread}")
    if (metric == "Time"):
        ax.set_ylabel("Runtime (s)")
    elif (metric == "Energy"):
        ax.set_ylabel("Energy (J)")
    # ax.set_xlabel("Ring Dimensions (2^n)")
    ax.grid(True)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.grid(which='major', linestyle='-', linewidth=0.7)
    ax.grid(which='minor', linestyle=':', linewidth=0.5)
    if set_legend:
        ax.legend(
            # title="Microbenchmarks",
            loc='lower center',
            bbox_to_anchor=(1.3, 1.2),
            fontsize=FONT_SIZE,
            title_fontsize=FONT_SIZE,
            ncols=6,
            handlelength=1.2,           # shorten legend line
            labelspacing=0.0,
            columnspacing=0.5
        )
    
    ax.tick_params(axis='y', labelcolor='black')
    # ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax.set_title(f"{tag}", fontsize=TITLE_SIZE, pad=3)
    
#Figure 10: Relative error of runtime and energy prediction on AMD cpu across thread counts. Each curve is the error rate at each thread count (3 microbenchmarks and four workloads)
def figure_10_relative_error_application_curve_thread_sweep(ax, tag, set_legend, platform: str, benchmarks: List[str], sec_std: str, n: int, depth: int, thread_counts: List[int], metric: str = "Time", output_dir: str = OUTPUT_DIR) -> None:
    estimated_values: Dict[str, Dict[int, float]] = {}
    actual_values: Dict[str, Dict[int, float]] = {}
    relative_percentage_errors: Dict[str, Dict[int, float]] = {}

    
    for benchmark in benchmarks:
        estimated_values[benchmark] = {}
        actual_values[benchmark] = {}
        relative_percentage_errors[benchmark] = {}
        
        csv_path = estimated_vs_actual_time_dict[platform]["estimated"][benchmark] if metric == "Time" else estimated_vs_actual_energy_dict[platform]["estimated"][benchmark]
        if csv_path is None:
            continue
        est_df = pd.read_csv(csv_path)
        
        csv_path = estimated_vs_actual_time_dict[platform]["actual"][benchmark] if metric == "Time" else estimated_vs_actual_energy_dict[platform]["actual"][benchmark]
        if csv_path is None:
            continue
        act_df = pd.read_csv(csv_path)
            
        for thread_count in thread_counts:
            filtered_est_df = est_df
            if thread_count is not None:
                filtered_est_df = filtered_est_df[filtered_est_df["num_threads"] == thread_count]
            if benchmark in ["matrix_multiplication", "logistic_function", "sign_eval"]:
                if sec_std is not None:
                    filtered_est_df = filtered_est_df[filtered_est_df["security_standard_level"] == sec_std]
                if n is not None:
                    filtered_est_df = filtered_est_df[filtered_est_df["n"] == n]
                if depth is not None:
                    filtered_est_df = filtered_est_df[filtered_est_df["depth"] == depth]
                    
            elif benchmark == "low_memory_resnet20":
                filtered_est_df = filtered_est_df[filtered_est_df["security_standard_level"] == "none"]
                filtered_est_df = filtered_est_df[filtered_est_df["n"] == 16]
                filtered_est_df = filtered_est_df[filtered_est_df["depth"] == 10]
                
            elif benchmark == "logistic_regression":
                filtered_est_df = filtered_est_df[filtered_est_df["security_standard_level"] == "128c"]
                filtered_est_df = filtered_est_df[filtered_est_df["depth"] == 14]
                
            elif benchmark == "cifar10":
                filtered_est_df = filtered_est_df[filtered_est_df["security_standard_level"] == "none"]
                filtered_est_df = filtered_est_df[filtered_est_df["n"] == 16]
                filtered_est_df = filtered_est_df[filtered_est_df["depth"] == 5] # Needs to be changed to 5
                
            elif benchmark == "chi_square_test":
                filtered_est_df = filtered_est_df[filtered_est_df["security_standard_level"] == "none"]
                filtered_est_df = filtered_est_df[filtered_est_df["n"] == 17]       # may be changed
                filtered_est_df = filtered_est_df[filtered_est_df["depth"] == 3] # needs to be changed to 3
                
            if len(filtered_est_df) == 0:
                raise ValueError(
                    f"No estimated values for {benchmark} with sec_std={sec_std}, n={n}, "
                    f"depth={depth}, num_threads={thread_count}"
                )
            
            estimated_values[benchmark][thread_count] = filtered_est_df[f"estimated_{metric}"].iloc[0]
            
            # Locate actual profiled values
            filtered_actual_df = act_df

            if thread_count is not None:
                filtered_actual_df = filtered_actual_df[filtered_actual_df["num_threads"] == thread_count]
            
            if benchmark in ["matrix_multiplication", "logistic_function", "sign_eval"]:
                if sec_std is not None:
                    filtered_actual_df = filtered_actual_df[filtered_actual_df["security_standard_level"] == sec_std]
                if n is not None:
                    filtered_actual_df = filtered_actual_df[filtered_actual_df["n"] == n]
                if depth is not None:
                    filtered_actual_df = filtered_actual_df[filtered_actual_df["depth"] == depth]
                
            if len(filtered_actual_df) == 0:
                raise ValueError(
                    f"No actual values for {benchmark} with sec_std={sec_std}, n={n}, "
                    f"depth={depth}, num_threads={thread_count}"
                )
            
            actual_values[benchmark][thread_count], num_samples = median_of_column(filtered_actual_df, benchmark, metric)
            
            # print(f"Estimated {metric} for {benchmark} with {thread_count} threads: {estimated_values[benchmark][thread_count]}")
            # print(f"Actual {metric} for {benchmark} with {thread_count} threads: {actual_values[benchmark][thread_count]}")
            
            relative_percentage_errors[benchmark][thread_count] = (estimated_values[benchmark][thread_count] - actual_values[benchmark][thread_count]) / actual_values[benchmark][thread_count] * 100
            
    # fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT * (2 if with_legend else 1)))
    
    # Create evenly spaced x positions
    x_positions = np.arange(len(thread_counts))
    
    for benchmark in benchmarks:
        y_values = [relative_percentage_errors[benchmark][tc] for tc in thread_counts]
        ax.plot(x_positions, y_values, label=application_label_map[benchmark])

    # Geometric mean groups
    microbenchmarks = ["matrix_multiplication", "sign_eval", "logistic_function"]
    workloads = ["cifar10", "low_memory_resnet20", "logistic_regression", "chi_square_test"]
    
    all_microbenchmark_errors = []
    all_workload_errors = []

    microbenchmark_mean_thread = []
    workload_mean_thread = []
    
    for thread_count in thread_counts:
        microbenchmarks_errors = [relative_percentage_errors[b][thread_count] for b in microbenchmarks if b in relative_percentage_errors and thread_count in relative_percentage_errors[b]]
        workloads_errors = [relative_percentage_errors[b][thread_count] for b in workloads if b in relative_percentage_errors and thread_count in relative_percentage_errors[b]]
        
        all_microbenchmark_errors.extend(microbenchmarks_errors)
        all_workload_errors.extend(workloads_errors)

        if microbenchmarks_errors:
            microbenchmark_mean_thread.append(
                np.power(np.prod(np.abs(microbenchmarks_errors)), 1 / len(microbenchmarks_errors))
                * np.sign(np.mean(microbenchmarks_errors))
            )
        else:
            microbenchmark_mean_thread.append(np.nan)

        if workloads_errors:
            workload_mean_thread.append(
                np.power(np.prod(np.abs(workloads_errors)), 1 / len(workloads_errors))
                * np.sign(np.mean(workloads_errors))
            )
        else:
            workload_mean_thread.append(np.nan)

    # per thread geomean
    ax.plot(x_positions, microbenchmark_mean_thread, color='black', linestyle='--', linewidth=2, label='Microbenchmark geomean')
    ax.plot(x_positions, workload_mean_thread, color='gray', linestyle=':', linewidth=2, label='Workload geomean')
    
    ax.set_ylabel("Relative Error (%)")
    if metric == "Time":
        ax.set_yticks(np.arange(-50, 50, 25))
    elif metric == "Energy":
        ax.set_yticks(np.arange(-50, 75, 25))
    ax.set_xticks(x_positions)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(thread_counts)
    if set_legend:
        ax.legend(
            # title="Metric",
            loc='lower center',
            bbox_to_anchor=(1.1, 1.1),
            ncol=2,
            fontsize=FONT_SIZE,
            title_fontsize=FONT_SIZE,
            handlelength=1.2,           # shorten legend line
            labelspacing=0.0,
            columnspacing=0.5
        )
    ax.grid(True)
    ax.set_title(f"{tag}", fontsize=TITLE_SIZE, pad=3)


# Figure 11: Breakdown for runtime and energy prediciton with 8 threads for microbenchmarks and workloads (x axis)
def figure_11_primitive_breakdown_application_sweep(ax, tag, set_legend, platform: str, sec_std: str, ring_dim: int, batch_size: int, depth: int, num_threads: int, benchmarks: List[str], primitives: List[str], metric: str = "Time", output_dir: str = OUTPUT_DIR) -> None:
    bar_width = 0.35
    x = np.arange(len(benchmarks))
    # fig, ax = plt.subplots(figsize=(FIG_WIDTH*2, FIG_HEIGHT * (2 if with_legend else 1)))

    estimated_values: Dict[str, Dict[str,float]] = {}
    actual_values: Dict[str, Dict[str,float]] = {}
    
    # contributions_dict = runtime_contributions_dict if metric == "Time" else energy_contributions_dict
    contributions_dict = runtime_contributions_dict
    
    # Process estimated values
    for benchmark, csv_path in contributions_dict["estimated"].items():
        if csv_path is None:
            continue
        est_df = pd.read_csv(csv_path)
        filtered_df = est_df
        
        if num_threads is not None:
            filtered_df = filtered_df[filtered_df["num_threads"] == num_threads]

        if benchmark in ["matrix_multiplication", "logistic_function", "sign_eval"]:
            if sec_std is not None:
                filtered_df = filtered_df[filtered_df["security_standard_level"] == sec_std]
            if ring_dim is not None:
                filtered_df = filtered_df[filtered_df["n"] == ring_dim]
            if depth is not None:
                filtered_df = filtered_df[filtered_df["depth"] == depth]
        
        if len(filtered_df) == 0:
            raise ValueError(
                f"No estimated values for {benchmark} with sec_std={sec_std}, n={ring_dim}, "
                f"depth={depth}, num_threads={num_threads}"
            )
            
        estimated_values[benchmark] = {}
        total_contribution = 0.0
        
        # Collect all contributions
        for prim in primitives_dict[benchmark]:
            col = f"{prim}_contribution"
            if col in filtered_df.columns:
                contribution = filtered_df[col].median()
                estimated_values[benchmark][prim] = contribution
                total_contribution += contribution
            else:
                estimated_values[benchmark][prim] = 0.0
        
        # Convert to percentages
        if total_contribution > 0:
            for prim in estimated_values[benchmark]:
                estimated_values[benchmark][prim] = (estimated_values[benchmark][prim] * 100.0) / total_contribution
        
        # Combine contributions of primitives with varients (i.e. EvalMult and EvalMult (Plaintext))
        for prim in list(estimated_values[benchmark].keys()):
            if "_Plaintext" in prim or "_Scalar" in prim:
                base_prim = prim.replace("_Plaintext", "").replace("_Scalar", "")
                if base_prim in estimated_values[benchmark]:
                    estimated_values[benchmark][base_prim] += estimated_values[benchmark][prim]
                else:
                    estimated_values[benchmark][base_prim] = estimated_values[benchmark][prim]
                del estimated_values[benchmark][prim]
        
        # Renormalize again
        total_contribution = sum(estimated_values[benchmark].values())
        if total_contribution > 0:
            for prim in estimated_values[benchmark]:
                estimated_values[benchmark][prim] = (estimated_values[benchmark][prim] * 100.0) / total_contribution
    
    # Process actual metrics (as opposed to estimated)
    for benchmark, csv_path in contributions_dict["actual"].items():
        if csv_path is None:
            continue
        actual_df = pd.read_csv(csv_path)
        actual_values[benchmark] = dict(zip(actual_df["Function"], actual_df["Percentage of Time (%)"]))


        total_contribution = sum(actual_values[benchmark].values())
        if total_contribution > 0:
            for prim in actual_values[benchmark]:
                actual_values[benchmark][prim] = (actual_values[benchmark][prim] * 100.0) / total_contribution
                
    all_unique_primitives = set()
    for benchmark in benchmarks:
        if benchmark in estimated_values:
            all_unique_primitives.update(estimated_values[benchmark].keys())
        if benchmark in actual_values:
            all_unique_primitives.update(actual_values[benchmark].keys())
    
    primitive_colors = {prim: plt.cm.tab20(i % 20) for i, prim in enumerate(sorted(all_unique_primitives))}

    if metric == "Energy":
        # if energy is specified, compare estimated runtime and estimated energy contributions
        # estimated_values is for runtime contributions
        # actual_values is for energy contributions
        # overwrite estimated_values with runtime contributions
        contributions_dict = energy_contributions_dict
        for benchmark, csv_path in contributions_dict["estimated"].items():
            if csv_path is None:
                continue
            est_df = pd.read_csv(csv_path)
            filtered_df = est_df
            
            if num_threads is not None:
                filtered_df = filtered_df[filtered_df["num_threads"] == num_threads]

            if benchmark in ["matrix_multiplication", "logistic_function", "sign_eval"]:
                if sec_std is not None:
                    filtered_df = filtered_df[filtered_df["security_standard_level"] == sec_std]
                if ring_dim is not None:
                    filtered_df = filtered_df[filtered_df["n"] == ring_dim]
                if depth is not None:
                    filtered_df = filtered_df[filtered_df["depth"] == depth]
            
            if len(filtered_df) == 0:
                raise ValueError(
                    f"No estimated values for {benchmark} with sec_std={sec_std}, n={ring_dim}, "
                    f"depth={depth}, num_threads={num_threads}"
                )
                
            actual_values[benchmark] = {}
            total_contribution = 0.0
            
            # Collect all contributions
            for prim in primitives_dict[benchmark]:
                col = f"{prim}_contribution"
                if col in filtered_df.columns:
                    contribution = filtered_df[col].median()
                    actual_values[benchmark][prim] = contribution
                    total_contribution += contribution
                else:
                    actual_values[benchmark][prim] = 0.0
            
            # Convert to percentages
            if total_contribution > 0:
                for prim in actual_values[benchmark]:
                    actual_values[benchmark][prim] = (actual_values[benchmark][prim] * 100.0) / total_contribution
            
            # Combine contributions of primitives with varients (i.e. EvalMult and EvalMult (Plaintext))
            for prim in list(actual_values[benchmark].keys()):
                if "_Plaintext" in prim or "_Scalar" in prim:
                    base_prim = prim.replace("_Plaintext", "").replace("_Scalar", "")
                    if base_prim in actual_values[benchmark]:
                        actual_values[benchmark][base_prim] += actual_values[benchmark][prim]
                    else:
                        actual_values[benchmark][base_prim] = actual_values[benchmark][prim]
                    del actual_values[benchmark][prim]
            
            # Renormalize again
            total_contribution = sum(actual_values[benchmark].values())
            if total_contribution > 0:
                for prim in actual_values[benchmark]:
                    actual_values[benchmark][prim] = (actual_values[benchmark][prim] * 100.0) / total_contribution

    legend_primitives = set()
    
    for i, benchmark in enumerate(benchmarks):
        est_vals = []
        est_primitives = []
        if benchmark in estimated_values:
            for prim in estimated_values[benchmark]:
                est_vals.append(estimated_values[benchmark][prim])
                mapped_prim = primitives_label_map.get(prim, prim)
                est_primitives.append(mapped_prim)

        act_vals = []
        act_primitives = []

        # left bar for estimated runtime values
        if benchmark in estimated_values:
            sorted_estimated_items = sorted(estimated_values[benchmark].items())
            bottom = 0
            for prim, val in sorted_estimated_items:
                if metric == "Time":
                    mapped_prim = primitives_label_map.get(prim, prim)
                else:
                    mapped_prim = prim
                # if benchmark == "matrix_multiplication":
                #     print(f"Primitive: {prim}, {primitive_colors[prim]}\n    Mapped primitive: {mapped_prim}, {primitive_colors[mapped_prim]}")
                label = None
                if mapped_prim not in legend_primitives:
                    label = mapped_prim
                    legend_primitives.add(mapped_prim)
                ax.bar(i-0.22, val, width=0.35, bottom=bottom,
                     label=label,
                     color=primitive_colors[primitives_label_map.get(prim, prim)]
                     )
                bottom += val

        # right bar for actual values/estimated energy values
        if benchmark in actual_values:
            sorted_actual_items = sorted(actual_values[benchmark].items())  # Alphabetical order by primitive name
            bottom = 0
            for prim, val in sorted_actual_items:
                # if benchmark == "matrix_multiplication":
                #     print(f"Primitive: {prim}, {primitive_colors[prim]}")
                label = None
                if prim not in legend_primitives:
                    label = primitives_label_map.get(prim, prim)
                    legend_primitives.add(prim)
                ax.bar(i + 0.22, val, width=0.35, bottom=bottom,
                        label=label,
                        color=primitive_colors[primitives_label_map.get(prim, prim)],
                        hatch='//')
        
                bottom += val

    ax.set_xticks(x)
    ax.set_xticklabels([application_label_map[b] for b in benchmarks], rotation=25, ha='right', fontsize=FONT_SIZE)
    ax.set_ylabel("Contribution (%)")
    ax.set_ylim(0, 102)
    if set_legend:
        ax.legend(
            # title="Primitives",
            loc='lower center',
            bbox_to_anchor=(0.425, 1.10),
            ncol=2,
            fontsize=FONT_SIZE,
            title_fontsize=FONT_SIZE,
            handlelength=1.2,           # shorten legend line
            labelspacing=0.0,
            columnspacing=0.5
        )

    ax.tick_params(axis='y', labelcolor='black')
    # ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax.set_title(f"{tag}", fontsize=TITLE_SIZE, pad=3)
    

def figure_12_estimated_improvements_thread_sweep(ax, tag, set_legend, platform: str, output_dir: str = OUTPUT_DIR) -> None:
    if platform == "AMD":
        time_original_df = pd.read_csv(AMD_TIME_ORIGINAL_ALGO_CSV)
        time_new_df = pd.read_csv(AMD_TIME_NEW_ALGO_CSV)
        energy_original_df = pd.read_csv(AMD_ENERGY_ORIGINAL_ALGO_CSV)
        energy_new_df = pd.read_csv(AMD_ENERGY_NEW_ALGO_CSV)
    else:
        time_original_df = pd.read_csv(INTEL_TIME_ORIGINAL_ALGO_CSV)
        time_new_df = pd.read_csv(INTEL_TIME_NEW_ALGO_CSV)
        energy_original_df = pd.read_csv(INTEL_ENERGY_ORIGINAL_ALGO_CSV)
        energy_new_df = pd.read_csv(INTEL_ENERGY_NEW_ALGO_CSV)
    
    time_original_filtered = time_original_df[
        (time_original_df["security_standard_level"] == "none") &
        (time_original_df["n"] == 16) &
        (time_original_df["depth"] == 6)
    ]
    time_new_filtered = time_new_df[
        (time_new_df["security_standard_level"] == "none") &
        (time_new_df["n"] == 16) &
        (time_new_df["depth"] == 3)
    ]
    energy_original_filtered = energy_original_df[
        (energy_original_df["security_standard_level"] == "none") &
        (energy_original_df["n"] == 16) &
        (energy_original_df["depth"] == 6)
    ]
    energy_new_filtered = energy_new_df[
        (energy_new_df["security_standard_level"] == "none") &
        (energy_new_df["n"] == 16) &
        (energy_new_df["depth"] == 3)
    ]
    
    # Get unique thread counts and sort them
    thread_counts = sorted(time_original_filtered["num_threads"].unique())
    
    time_ratios = []
    energy_ratios = []
    
    for threads in thread_counts:
        time_orig = time_original_filtered[time_original_filtered["num_threads"] == threads]["estimated_Time"].iloc[0]
        time_new = time_new_filtered[time_new_filtered["num_threads"] == threads]["estimated_Time"].iloc[0]
        energy_orig = energy_original_filtered[energy_original_filtered["num_threads"] == threads]["estimated_Energy"].iloc[0]
        energy_new = energy_new_filtered[energy_new_filtered["num_threads"] == threads]["estimated_Energy"].iloc[0]
        
        time_ratios.append(time_orig / time_new)
        energy_ratios.append(energy_orig / energy_new)
    
    # fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT * (2 if with_legend else 1)))
    
    x = np.arange(len(thread_counts))
    width = 0.25  # Reduced from 0.35 to create separation
    
    time_bars = ax.bar(x - width/2 - 0.05, time_ratios, width, label='Time', color='#FF6B6B', alpha=0.8)
    energy_bars = ax.bar(x + width/2 + 0.05, energy_ratios, width, label='Energy', color='#4ECDC4', alpha=0.8)
    
    ax.set_ylabel('Improvement', fontsize=FONT_SIZE)
    ax.set_xticks(x)
    
    labels = [str(tc) if i % 2 == 0 else '' for i, tc in enumerate(thread_counts)]
    ax.set_xticklabels(labels)
    if set_legend:
        ax.legend(
            # title="Microbenchmarks",
            loc='lower center',
            bbox_to_anchor=(1.3, 1.2),
            fontsize=FONT_SIZE,
            title_fontsize=FONT_SIZE,
            ncols=2,
            handlelength=1.2,           # shorten legend line
            labelspacing=0.0,
            columnspacing=0.5
        )
    
    ax.tick_params(axis='y', labelcolor='black')
    # ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax.set_title(f"{tag} {platform}", fontsize=TITLE_SIZE, pad=3)

    ax.grid(True, alpha=0.3)
    

def filter_given_params(df: pd.DataFrame, security_standard: Optional[str], n: Optional[int], batch_size: Optional[int], 
    depth: Optional[int], num_threads: Optional[int]) -> pd.DataFrame:
    filtered_df = df
    if security_standard is not None:
        filtered_df = filtered_df[filtered_df["security_standard_level"] == security_standard]
    if n is not None:
        filtered_df = filtered_df[filtered_df["n"] == n]
    if batch_size is not None:
        filtered_df = filtered_df[filtered_df["batch_size"] == batch_size]
    if depth is not None:
        filtered_df = filtered_df[filtered_df["depth"] == depth]
    if num_threads is not None:
        filtered_df = filtered_df[filtered_df["num_threads"] == num_threads]

    if len(filtered_df) == 0:
        raise ValueError("No data found for the given parameters.")
    
    return filtered_df


def group_by_column(df: pd.DataFrame, group_by: str, target: str, metric: str = "Time", statistic = "median") -> pd.DataFrame:
    """Groups the DataFrame by a specified column and returns the mean of the target column."""
    column_name = f"{target} {metric}"
    if column_name in df.columns:
        if statistic == "median":
            grouped_df = df.groupby(group_by)[column_name].median().reset_index()
        elif statistic == "mean":
            grouped_df = df.groupby(group_by)[column_name].mean().reset_index()
        return grouped_df
    else:
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")
    

def median_of_column(df: pd.DataFrame, target: str, metric: str = "Time") -> Tuple[float, int]:
    """Returns the median of a specified column in a DataFrame and the number of rows in the DataFrame."""
    column_name = f"{target} {metric}"
    if column_name in df.columns:
        median_val = df[column_name].median()
        num_rows = len(df)
        return median_val, num_rows
    else:
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")


def print_figure_status(figure_label: str) -> None:
    print(f"Printed Figure {figure_label}", flush=True)


def save_figure(output_dir: Path, file_name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / file_name, bbox_inches="tight", pad_inches=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate ISPASS plots used by the artifact evaluation package."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Directory for generated PDFs (default: {OUTPUT_DIR})",
    )
    return parser.parse_args()


def main(output_dir: Path) -> None:
    amd_microbench_df = pd.read_csv(AMD_MICROBENCH_CSV)
    amd_microbench_matmul_sizes_df = pd.read_csv(AMD_MICROBENCH_MATMUL_SIZES_CSV)
    amd_primitive_runtime_df = pd.read_csv(AMD_PRIMITIVE_RUNTIME_CSV)
    amd_primitive_event_df = pd.read_csv(AMD_PRIMITIVE_EVENT_CSV)
    amd_primitives_hardware_backend_df = pd.read_csv(AMD_PRIMITIVES_HARDWARE_BACKEND_CSV)
    intel_microbench_no_sec_std_df = pd.read_csv(INTEL_MICROBENCH_NO_SEC_STD_CSV)
    intel_primitive_no_sec_std_runtime_df = pd.read_csv(INTEL_PRIMITIVE_RUNTIME_CSV)

    primitives = ["CKKS_Add", "CKKS_Add_Plaintext", "CKKS_Mult", "CKKS_Mult_Plaintext", "CKKS_Rotate"]
    microbenchmarks = ["matrix_multiplication", "logistic_function", "sign_eval"]
    workloads = ["cifar10", "low_memory_resnet20", "logistic_regression", "chi_square_test"]
    benchmarks = microbenchmarks + workloads
    thread_counts = [1, 2, 4, 8, 16, 32]
    ring_dimensions = [13,14,15,16,17]
    sec_std = "none"
    ring_dim = 16
    depth = 10
    batch_size = 12
    
    plt.rcParams.update({'font.size': FONT_SIZE})

    # Figures 5a-5d
    figure_5_df_amd = filter_given_params(amd_primitive_runtime_df, sec_std, ring_dim, batch_size, depth, None)
    figure_5_df_intel = filter_given_params(intel_primitive_no_sec_std_runtime_df, sec_std, ring_dim, batch_size, depth, None)
    fig = plt.figure(figsize=(FIG_WIDTH*2, FIG_HEIGHT*1.5))
    ax1 = fig.add_axes([0.0, 1.0, 0.27, 0.3])
    ax2 = fig.add_axes([0.0, 0.5, 0.27, 0.3])
    ax3 = fig.add_axes([0.5, 1.0, 0.27, 0.3])
    ax4 = fig.add_axes([0.5, 0.5, 0.27, 0.3])
    figure_5_primitive_curve_thread_sweep(ax1, "(a)", True, figure_5_df_amd, "AMD", primitives, "Time")
    figure_5_primitive_curve_thread_sweep(ax2, "(c)", False, figure_5_df_amd, "AMD", primitives, "Energy")
    figure_5_primitive_curve_thread_sweep(ax3, "(b)", False, figure_5_df_intel, "Intel", primitives, "Time")
    figure_5_primitive_curve_thread_sweep(ax4, "(d)", False, figure_5_df_intel, "Intel", primitives, "Energy")
    save_figure(output_dir, "figure_5_ad.pdf")
    plt.close()
    print_figure_status("5")
    
    # Figures 6a-6d
    figure_6_df_amd = filter_given_params(amd_primitive_event_df, sec_std, ring_dim, batch_size, depth, None)
    fig = plt.figure(figsize=(FIG_WIDTH*2, FIG_HEIGHT*1.5))
    ax1 = fig.add_axes([0.0, 1.0, 0.27, 0.3])
    ax2 = fig.add_axes([0.0, 0.5, 0.27, 0.3])
    ax3 = fig.add_axes([0.5, 1.0, 0.27, 0.3])
    ax4 = fig.add_axes([0.5, 0.5, 0.27, 0.3])
    figure_6_primitive_curve_event_stats_thread_sweep(ax1, "(a)", True, figure_6_df_amd, "AMD", primitives, "instructions")
    figure_6_primitive_curve_event_stats_thread_sweep(ax2, "(c)", False, figure_6_df_amd, "AMD", primitives, "cache-references")
    figure_6_primitive_curve_event_stats_thread_sweep(ax3, "(b)", False, figure_6_df_amd, "AMD", primitives, "dTLB-loads")
    figure_6_primitive_curve_event_stats_thread_sweep(ax4, "(d)", False, figure_6_df_amd, "AMD", primitives, "page-faults")
    save_figure(output_dir, "figure_6_ad.pdf")
    plt.close()
    print_figure_status("6")

    # Figures 7a-7d
    figure_7_df_amd = filter_given_params(amd_microbench_df, sec_std, ring_dim, batch_size, depth, None)
    figure_7_df_intel = filter_given_params(intel_microbench_no_sec_std_df, sec_std, ring_dim, batch_size, depth, None)
    fig = plt.figure(figsize=(FIG_WIDTH*2, FIG_HEIGHT*1.5))
    ax1 = fig.add_axes([0.0, 1.0, 0.27, 0.3])
    ax2 = fig.add_axes([0.0, 0.5, 0.27, 0.3])
    ax3 = fig.add_axes([0.5, 1.0, 0.27, 0.3])
    ax4 = fig.add_axes([0.5, 0.5, 0.27, 0.3])
    figure_7_microbenchmark_curve_thread_sweep(ax1, "(a)", True, figure_7_df_amd, "AMD", microbenchmarks, "Time")
    figure_7_microbenchmark_curve_thread_sweep(ax2, "(c)", False, figure_7_df_amd, "AMD", microbenchmarks, "Energy")
    figure_7_microbenchmark_curve_thread_sweep(ax3, "(b)", False, figure_7_df_intel, "Intel", microbenchmarks, "Time")
    figure_7_microbenchmark_curve_thread_sweep(ax4, "(d)", False, figure_7_df_intel, "Intel", microbenchmarks, "Energy")
    save_figure(output_dir, "figure_7_ad.pdf")
    plt.close()
    print_figure_status("7")
    
    # Figures 8a, 8b
    thread_count = 1
    figure_8_df_amd = filter_given_params(amd_primitives_hardware_backend_df, sec_std, None, batch_size, depth, thread_count)
    fig = plt.figure(figsize=(FIG_WIDTH*2, FIG_HEIGHT*1.5))
    ax1 = fig.add_axes([0.5, 0.0, 0.3, 0.4])
    ax2 = fig.add_axes([1.0, 0.0, 0.3, 0.4])
    primitives = ["CKKS_Mult"]
    figure_8_primitive_hardware_backend_curve_ring_sweep(ax1, "Runtime", False, figure_8_df_amd, "AMD", primitives, "Time")
    primitives = ["CKKS_Mult_Plaintext"]
    figure_8_primitive_hardware_backend_curve_ring_sweep(ax2, "Energy", False, figure_8_df_amd, "AMD", primitives, "Energy")
    save_figure(output_dir, "figure_8_ab.pdf")
    plt.close()
    print_figure_status("8")
    
    # Figures 9a-9d
    batch_size = 13
    figure_9_df = filter_given_params(amd_microbench_matmul_sizes_df, "none", None, batch_size, depth, None)
    fig = plt.figure(figsize=(FIG_WIDTH*2, FIG_HEIGHT*1.5))
    ax1 = fig.add_axes([0.0, 1.0, 0.27, 0.3])
    ax2 = fig.add_axes([0.0, 0.5, 0.27, 0.3])
    ax3 = fig.add_axes([0.5, 1.0, 0.27, 0.3])
    ax4 = fig.add_axes([0.5, 0.5, 0.27, 0.3])
    figure_9_matmul_thread_curve_ring_sweep(ax1, "(a) 8x8 by 8x8", True, figure_9_df, "AMD", 8, ring_dimensions[1:], thread_counts)
    figure_9_matmul_thread_curve_ring_sweep(ax2, "(c) 32x32 by 32x32", False, figure_9_df, "AMD", 32, ring_dimensions[1:], thread_counts)
    figure_9_matmul_thread_curve_ring_sweep(ax3, "(b) 16x16 by 16x16", False, figure_9_df, "AMD", 16, ring_dimensions[1:], thread_counts)
    figure_9_matmul_thread_curve_ring_sweep(ax4, "(d) 64x64 by 64x64", False, figure_9_df, "AMD", 64, ring_dimensions[1:], thread_counts)
    save_figure(output_dir, "figure_9_ad.pdf")
    plt.close()
    print_figure_status("9")

    # Figure 10
    fig = plt.figure(figsize=(FIG_WIDTH*2, FIG_HEIGHT*1.5))
    ax1 = fig.add_axes([0.5, 0.0, 0.3, 0.4])
    ax2 = fig.add_axes([1.0, 0.0, 0.3, 0.4])
    figure_10_relative_error_application_curve_thread_sweep(ax1, "Runtime", True, "AMD", benchmarks, "none", 16, 20, thread_counts, "Time")
    figure_10_relative_error_application_curve_thread_sweep(ax2, "Energy", False, "AMD", benchmarks, "none", 16, 20, thread_counts, "Energy")
    save_figure(output_dir, "figure_10_ab.pdf")
    plt.close()
    print_figure_status("10")

    # Figures 11a, 11b
    num_threads = 8
    fig = plt.figure(figsize=(FIG_WIDTH*2, FIG_HEIGHT*1.8))
    ax1 = fig.add_axes([0.0, 1.0, 1, 0.38])
    ax2 = fig.add_axes([0.0, 0.4, 1, 0.38])
    figure_11_primitive_breakdown_application_sweep(ax1, "(a) Predicted runtime (left) vs actual runtime (right).", True, "AMD", sec_std, ring_dim, batch_size, depth, num_threads, benchmarks, primitives, metric="Time")
    figure_11_primitive_breakdown_application_sweep(ax2, "(b) Predicted runtime (left) vs predicted energy (right)", False, "AMD", sec_std, ring_dim, batch_size, depth, num_threads, benchmarks, primitives, metric="Energy")
    save_figure(output_dir, "figure_11_ab.pdf")
    plt.close()
    print_figure_status("11")

    # Figure 12
    fig = plt.figure(figsize=(FIG_WIDTH*2, FIG_HEIGHT*1))
    ax1 = fig.add_axes([0.0, 1.0, 0.27, 0.3])
    ax2 = fig.add_axes([0.5, 1.0, 0.27, 0.3])
    figure_12_estimated_improvements_thread_sweep(ax1, "(a)", True, "AMD")
    figure_12_estimated_improvements_thread_sweep(ax2, "(b)", False, "Intel")
    save_figure(output_dir, "figure_12.pdf")
    plt.close()
    print_figure_status("12")


if __name__ == "__main__":
    args = parse_args()
    main(args.output_dir)

    
