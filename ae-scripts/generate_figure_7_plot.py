"""Recreate Figure 7 from the paper using profiled microbenchmark results."""

import argparse
from pathlib import Path
from typing import List, Optional
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATASETS_DIR = REPO_ROOT / "ae-datasets"
OUTPUT_DIR = REPO_ROOT / "ae-plots"

DEFAULT_MICROBENCH_CSV = REPO_ROOT / "out/csv/microbenchmark-results-ae.csv"
DEFAULT_OUTPUT_FILENAME = "figure_7_plot.pdf"

# Figure dimension/style parameters matching generate-ispass-plots.py
BASE_TEXTWIDTH = 6.5
FIG_WIDTH = BASE_TEXTWIDTH * 0.25
FIG_HEIGHT = BASE_TEXTWIDTH * 0.25
FONT_SIZE = 8
TITLE_SIZE = 8

# Fixed Figure 7 parameters from the paper setup.
FIG7_SECURITY_STANDARD_LEVEL = "none"
FIG7_RING_DIM = 16
FIG7_BATCH_SIZE = 12
FIG7_DEPTH = 10
FIG7_MICROBENCHMARKS = ["matrix_multiplication", "logistic_function", "sign_eval"]


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


def figure_7_microbenchmark_curve_thread_sweep(
    ax: plt.Axes,
    tag: str,
    set_legend: bool,
    df: pd.DataFrame,
    microbenchmarks: List[str],
    metric: str = "Time",
) -> None:
    for microbenchmark in microbenchmarks:
        grouped_df = group_by_column(df, "num_threads", microbenchmark, metric)
        y = grouped_df[f"{microbenchmark} {metric}"]
        ax.plot(grouped_df["num_threads"].astype(str), y, label=microbenchmark)

    if metric == "Time":
        ax.set_ylabel("Runtime (s)")
    elif metric == "Energy":
        ax.set_ylabel("Energy (J)")

    ax.grid(True)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    if set_legend:
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(1.3, 1.3),
            fontsize=FONT_SIZE,
            title_fontsize=FONT_SIZE,
            ncols=2,
            handlelength=1.2,
            labelspacing=0.0,
            columnspacing=0.5,
        )

    ax.tick_params(axis="y", labelcolor="black")
    ax.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
    ax.set_title(f"{tag}", fontsize=TITLE_SIZE, pad=3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a Figure 7 runtime+energy microbenchmark plot."
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=DEFAULT_MICROBENCH_CSV,
        help="Microbenchmark CSV path.",
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
        FIG7_SECURITY_STANDARD_LEVEL,
        FIG7_RING_DIM,
        FIG7_BATCH_SIZE,
        FIG7_DEPTH,
    )

    plt.rcParams.update({"font.size": FONT_SIZE})
    fig = plt.figure(figsize=(FIG_WIDTH * 2, FIG_HEIGHT * 1.5))
    ax_runtime = fig.add_axes([0.5, 0.0, 0.3, 0.4])
    ax_energy = fig.add_axes([1.0, 0.0, 0.3, 0.4])

    figure_7_microbenchmark_curve_thread_sweep(
        ax_runtime, "(a) Runtime", True, df, FIG7_MICROBENCHMARKS, "Time"
    )
    figure_7_microbenchmark_curve_thread_sweep(
        ax_energy, "(b) Energy", False, df, FIG7_MICROBENCHMARKS, "Energy"
    )

    output_path = args.output_dir / DEFAULT_OUTPUT_FILENAME
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved plot: {output_path}")


if __name__ == "__main__":
    main()
