"""Recreate Figure 5 from the paper using profiled OpenFHE primitive results."""

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATASETS_DIR = REPO_ROOT / "ae-datasets"
OUTPUT_DIR = REPO_ROOT / "ae-plots"

DEFAULT_PRIMITIVE_RUNTIME_CSV = REPO_ROOT / "out/csv/primitive-results-ae.csv"
DEFAULT_OUTPUT_FILENAME = "figure_5_plot.pdf"

# Fixed Figure 5 parameters from the paper setup.
FIG5_SECURITY_STANDARD_LEVEL = "none"
FIG5_RING_DIM = 16
FIG5_BATCH_SIZE = 12
FIG5_DEPTH = 10

DEFAULT_PRIMITIVES = [
    "CKKS_Add",
    "CKKS_Add_Plaintext",
    "CKKS_Mult",
    "CKKS_Mult_Plaintext",
    "CKKS_Rotate",
]

PRIMITIVES_LABEL_MAP = {
    "CKKS_Add": "EvalAdd",
    "CKKS_Add_Plaintext": "EvalAdd (Ptxt)",
    "CKKS_Mult": "EvalMult",
    "CKKS_Mult_Plaintext": "EvalMult (Ptxt)",
    "CKKS_Rotate": "EvalRotate",
}

# Figure dimension/style parameters matching generate-ispass-plots.py
BASE_TEXTWIDTH = 6.5
FIG_WIDTH = BASE_TEXTWIDTH * 0.25
FIG_HEIGHT = BASE_TEXTWIDTH * 0.25
FONT_SIZE = 8
TITLE_SIZE = 8


def filter_given_params(
    df: pd.DataFrame,
    security_standard: Optional[str],
    n: Optional[int],
    batch_size: Optional[int],
    depth: Optional[int],
) -> pd.DataFrame:
    filtered_df = df
    if security_standard is not None:
        filtered_df = filtered_df[filtered_df["security_standard_level"] == security_standard]
    if n is not None:
        filtered_df = filtered_df[filtered_df["n"] == n]
    if batch_size is not None:
        filtered_df = filtered_df[filtered_df["batch_size"] == batch_size]
    if depth is not None:
        filtered_df = filtered_df[filtered_df["depth"] == depth]
    if filtered_df.empty:
        raise ValueError("No data found for the requested filter parameters.")
    return filtered_df


def group_by_column(df: pd.DataFrame, group_by: str, target: str, metric: str = "Time") -> pd.DataFrame:
    column_name = f"{target} {metric}"
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the CSV.")
    return df.groupby(group_by)[column_name].median().reset_index().sort_values(group_by)


def plot_primitive_curve_thread_sweep(
    ax1: plt.Axes,
    df: pd.DataFrame,
    primitives: List[str],
    metric: str,
    title: str,
    set_legend: bool,
) -> Tuple[List[plt.Line2D], List[str]]:
    ax2 = ax1.twinx()

    all_y = []
    for primitive in primitives:
        grouped_df = group_by_column(df, "num_threads", primitive, metric)
        y = grouped_df[f"{primitive} {metric}"]
        if metric == "Time":
            y = y * 1000  # seconds -> milliseconds
        all_y.extend(y.values)

    threshold = np.mean(all_y)
    idx1 = 0
    idx2 = 0
    colors1 = plt.cm.cool(np.linspace(0, 1, len(primitives)))
    colors2 = plt.cm.autumn(np.linspace(0, 0.8, 2))

    for primitive in primitives:
        grouped_df = group_by_column(df, "num_threads", primitive, metric)
        y = grouped_df[f"{primitive} {metric}"]
        if metric == "Time":
            y = y * 1000
        x = grouped_df["num_threads"].astype(str)
        label = PRIMITIVES_LABEL_MAP.get(primitive, primitive)
        if y.max() > threshold:
            ax2.plot(x, y, label=label, color=colors2[idx2], linestyle="--")
            idx2 += 1
        else:
            ax1.plot(x, y, label=label, color=colors1[idx1])
            idx1 += 1

    ax1.set_ylabel("Runtime (ms)" if metric == "Time" else "Energy (J)", fontsize=FONT_SIZE)
    ax1.tick_params(axis="y", labelcolor="black")
    ax2.tick_params(axis="y", labelcolor="black")
    ax1.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
    ax2.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
    ax1.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
    ax1.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax2.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
    ax2.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax1.grid(True)
    ax1.set_title(title, fontsize=TITLE_SIZE, pad=3)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    if set_legend:
        ax1.legend(
            lines1 + lines2,
            labels1 + labels2,
            loc="lower center",
            bbox_to_anchor=(1.3, 1.3),
            fontsize=FONT_SIZE,
            title_fontsize=FONT_SIZE,
            ncols=3,
            handlelength=1.2,
            labelspacing=0.0,
            columnspacing=0.5,
        )
    return lines1 + lines2, labels1 + labels2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Figure 5 runtime+energy plot for 5 primitives.")
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=DEFAULT_PRIMITIVE_RUNTIME_CSV,
        help="Primitive runtime CSV path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory where the plot PDF will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv_path)
    df = filter_given_params(
        df,
        FIG5_SECURITY_STANDARD_LEVEL,
        FIG5_RING_DIM,
        FIG5_BATCH_SIZE,
        FIG5_DEPTH,
    )

    plt.rcParams.update({"font.size": FONT_SIZE})
    fig = plt.figure(figsize=(FIG_WIDTH * 2, FIG_HEIGHT * 1.5))
    ax_runtime = fig.add_axes([0.5, 0.0, 0.3, 0.4])
    ax_energy = fig.add_axes([1.0, 0.0, 0.3, 0.4])

    plot_primitive_curve_thread_sweep(
        ax_runtime, df, DEFAULT_PRIMITIVES, "Time", "(a) Runtime", True
    )
    plot_primitive_curve_thread_sweep(
        ax_energy, df, DEFAULT_PRIMITIVES, "Energy", "(b) Energy", False
    )

    output_path = args.output_dir / DEFAULT_OUTPUT_FILENAME
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved plot: {output_path}")


if __name__ == "__main__":
    main()
