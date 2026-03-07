import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns
from pathlib import Path
import sys
import argparse
import yaml
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.append(str(Path(__file__).parent.parent))
from src.logging_utils import (
    print_argument,
    print_cryptoracle_banner,
    print_error,
    print_section_header,
    print_status,
    print_timestamp,
)

BASE_TEXTWIDTH = 6.5
FIG_WIDTH = BASE_TEXTWIDTH * 0.50
FIG_HEIGHT = BASE_TEXTWIDTH * 0.30
FONT_SIZE = 8
TITLE_SIZE = 8
TICK_SIZE = 8
LEGEND_SIZE = 7


def apply_paper_plot_style() -> None:
    """Apply compact, paper-style plotting defaults."""
    plt.rcParams.update(
        {
            "font.size": FONT_SIZE,
            "axes.titlesize": TITLE_SIZE,
            "axes.labelsize": FONT_SIZE,
            "xtick.labelsize": TICK_SIZE,
            "ytick.labelsize": TICK_SIZE,
            "legend.fontsize": LEGEND_SIZE,
        }
    )


def save_plot(fig: plt.Figure, path: Path) -> None:
    """Save plots with paper-like tight spacing."""
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Analyze application runtime using additive model from primitive benchmark results.')
    parser.add_argument('--config', '-c', type=str, 
                        required=True,
                        help='Path to the YAML configuration file')
    parser.add_argument('--output-dir', '-o', type=str,
                        required=True,
                        help='Directory to save output plots and CSV files')
    return parser.parse_args()


def print_cli_summary(args: argparse.Namespace) -> None:
    """Print command line inputs using CryptOracle terminal formatting."""
    print_section_header("Performance Model Inputs")
    print_argument("Config File:", str(Path(args.config).expanduser()))
    print_argument("Output Directory:", str(Path(args.output_dir).expanduser()))
    print()


def print_model_summary(config: dict) -> None:
    """Print key model settings from config."""
    print_section_header("Model Configuration")
    print_argument("Application:", str(config["name"]))
    print_argument("Statistic:", str(config["statistic"]))
    print_argument("Primitive CSV:", str(config["primitive_benchmarks"]))
    print_argument("Actual CSV:", str(config["actual_benchmarks"]))
    print_argument("Operation Count:", str(len(config["operation_counts"])))
    print()


def print_estimation_breakdown(estimated_df: pd.DataFrame, operation_counts: dict) -> None:
    """Print per-configuration data-point counts and estimated runtime."""
    print_section_header("Estimation Breakdown")
    for _, row in estimated_df.iterrows():
        print_argument("Security Level:", str(row["security_standard_level"]))
        print_argument("N:", str(int(row["n"])))
        for operation in operation_counts.keys():
            count_col = f"{operation}_count"
            if count_col in row:
                print_argument(f"{operation} Samples:", str(int(row[count_col])))
        print_argument("Estimated Runtime (s):", f"{row['estimated_time']:.3f}")
        print()


def load_config(config_path):
    """Load and validate configuration from YAML file."""
    try:
        config_path = Path(config_path).expanduser().resolve()
        with config_path.open('r') as f:
            config = yaml.safe_load(f)
        
        required_fields = ['name', 'primitive_benchmarks', 'actual_benchmarks', 
                         'statistic', 'title', 'operation_counts']
        
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field '{field}' in configuration")
        
        if config['statistic'] not in ['mean', 'median']:
            raise ValueError("statistic must be either 'mean' or 'median'")

        # Resolve CSV paths relative to the config file location.
        # Absolute paths are preserved as-is.
        config_dir = config_path.parent
        for field in ['primitive_benchmarks', 'actual_benchmarks']:
            field_path = Path(config[field]).expanduser()
            if not field_path.is_absolute():
                field_path = (config_dir / field_path).resolve()
            config[field] = str(field_path)
        
        return config
    except Exception as e:
        print_error(f"Error loading configuration: {e}")
        sys.exit(1)

def load_primitive_results(csv_path, config):
    """Load and process primitive benchmark results from CSV file."""
    try:
        df = pd.read_csv(csv_path)
        operations_data = []
        
        for operation in config['operation_counts'].keys():
            col_name = f"{operation} Time"
            if col_name not in df.columns:
                print_error(f"Column {col_name} not found in primitive benchmarks CSV")
                continue
                
            op_data = df[['security_standard_level', 'n', col_name]].copy()
            op_data['operation'] = operation
            op_data.rename(columns={col_name: 'time'}, inplace=True)
            operations_data.append(op_data)

        if not operations_data:
            raise ValueError("No valid operations found in the primitive benchmarks CSV")

        operations_df = pd.concat(operations_data)
        operations_df['time'] = operations_df['time']
        
        return operations_df
        
    except Exception as e:
        print_error(f"Error loading primitive results: {e}")
        sys.exit(1)

def calculate_estimated_runtime(primitive_df, operation_counts, statistic='mean'):
    """Calculate estimated runtime for each security standard and n value combination."""
    if statistic == 'mean':
        grouped = primitive_df.groupby(['security_standard_level', 'n', 'operation'])['time'].mean().reset_index()
    else:
        grouped = primitive_df.groupby(['security_standard_level', 'n', 'operation'])['time'].median().reset_index()

    counts = primitive_df.groupby(['security_standard_level', 'n', 'operation']).size().reset_index(name='count')
    
    grouped = grouped.merge(counts, on=['security_standard_level', 'n', 'operation'])

    pivoted = grouped.pivot(
        index=['security_standard_level', 'n'],
        columns='operation',
        values='time'
    ).reset_index()
    
    count_pivoted = counts.pivot(
        index=['security_standard_level', 'n'],
        columns='operation',
        values='count'
    ).reset_index()
    
    count_pivoted.columns = [f'{col}_count' if col in operation_counts.keys() else col 
                            for col in count_pivoted.columns]

    pivoted = pivoted.merge(count_pivoted, on=['security_standard_level', 'n'])
  
    estimated_time = 0
    for operation, count in operation_counts.items():
        if operation in pivoted.columns:
            estimated_time += count * pivoted[operation]
            pivoted[f'{operation}_contribution'] = count * pivoted[operation]
    
    pivoted['estimated_time'] = estimated_time
    return pivoted

def plot_estimated_runtimes(results_df, config, output_dir):
    """Create plots of estimated runtimes."""
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(FIG_WIDTH * 1.4, FIG_HEIGHT * 2.2))
    
    sec_standards_data = results_df[
        (results_df['security_standard_level'].isin(['128c', '192c', '256c'])) & 
        (results_df['n'] == 12)
    ]

    sns.barplot(data=sec_standards_data, x='security_standard_level', y='estimated_time', ax=ax1)
    ax1.set_title(f'Estimated {config["name"]} Runtime by Security Standard ({config["statistic"].capitalize()})')
    ax1.set_ylabel('Estimated Time (s)')
    ax1.set_xlabel('Security Standard Level')
    
    contribution_cols = [f'{op}_contribution' for op in config['operation_counts'].keys()]
    plot_data_sec = sec_standards_data.melt(
        id_vars=['security_standard_level'],
        value_vars=contribution_cols,
        var_name='operation',
        value_name='time'
    )
    sns.barplot(data=plot_data_sec, x='security_standard_level', y='time', hue='operation', ax=ax2)
    ax2.set_title(f'Operation Type Contributions by Security Standard ({config["statistic"].capitalize()})')
    ax2.set_ylabel('Time Contribution (s)')
    ax2.set_xlabel('Security Standard Level')
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    security_plot_path = output_dir / f"estimated_{config['title']}_security_{config['statistic']}.png"
    for axis in (ax1, ax2):
        axis.grid(True, alpha=0.3)
        axis.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
        axis.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        axis.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    save_plot(fig1, security_plot_path)
    print_status(f"Security standards comparison saved to {security_plot_path}")
    
    fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(FIG_WIDTH * 1.4, FIG_HEIGHT * 2.2))
 
    n_values_data = results_df[results_df['security_standard_level'] == 'none']
    
    sns.barplot(data=n_values_data, x='n', y='estimated_time', ax=ax3)
    ax3.set_title(f'Estimated {config["name"]} Runtime by n (security_standard_level = none, {config["statistic"].capitalize()})')
    ax3.set_ylabel('Estimated Time (s)')
    ax3.set_xlabel('n')

    plot_data_n = n_values_data.melt(
        id_vars=['n'],
        value_vars=contribution_cols,
        var_name='operation',
        value_name='time'
    )
    sns.barplot(data=plot_data_n, x='n', y='time', hue='operation', ax=ax4)
    ax4.set_title(f'Operation Type Contributions by n (security_standard_level = none, {config["statistic"].capitalize()})')
    ax4.set_ylabel('Time Contribution (s)')
    ax4.set_xlabel('n')
    
    n_plot_path = output_dir / f"estimated_{config['title']}_n_{config['statistic']}.png"
    for axis in (ax3, ax4):
        axis.grid(True, alpha=0.3)
        axis.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
        axis.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        axis.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    save_plot(fig2, n_plot_path)
    print_status(f"N values comparison saved to {n_plot_path}")

def load_and_process_data(actual_csv, estimated_csv, config):
    estimated_df = pd.read_csv(estimated_csv)
    actual_df = pd.read_csv(actual_csv)
    
    actual_df = actual_df[['security_standard_level', 'n', f'{config["name"]} Time']]
    actual_df = actual_df.rename(columns={f'{config["name"]} Time': 'actual_time'})
    
    actual_df = actual_df.groupby(['security_standard_level', 'n'])['actual_time'].mean().reset_index()
    
    merged_df = pd.merge(estimated_df, actual_df, on=['security_standard_level', 'n'])
    
    merged_df['relative_error'] = abs(merged_df['estimated_time'] - merged_df['actual_time']) / merged_df['actual_time'] * 100
    
    return merged_df

def calculate_statistics(df):
    mae = mean_absolute_error(df['actual_time'], df['estimated_time'])
    mse = mean_squared_error(df['actual_time'], df['estimated_time'])
    rmse = np.sqrt(mse)
    
    if len(df) > 1:
        r2 = r2_score(df['actual_time'], df['estimated_time'])
    else:
        r2 = float('nan')
    
    mean_relative_error = df['relative_error'].mean()
    
    stats = {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'Mean Relative Error (%)': mean_relative_error,
        'Number of Samples': len(df)
    }
    
    return stats

def print_reduced_statistics(stats: dict, indent: str = "") -> None:
    """Print only the metrics used in artifact-evaluation reporting."""
    num_samples = int(stats["Number of Samples"])
    mean_relative_error = stats["Mean Relative Error (%)"]
    if indent:
        print(f"{indent}Number of Samples: {num_samples}")
    else:
        print_argument("Number of Samples:", str(num_samples))
    if pd.isna(mean_relative_error):
        if indent:
            print(f"{indent}Mean Relative Error (%): undefined (needs more samples)")
        else:
            print_argument("Mean Relative Error (%):", "undefined (needs more samples)")
    else:
        if indent:
            print(f"{indent}Mean Relative Error (%): {mean_relative_error:.4f}")
        else:
            print_argument("Mean Relative Error (%):", f"{mean_relative_error:.4f}")

def plot_comparison(df, config, output_dir):
    df = df[df['n'] != 12]

    fig, ax = plt.subplots(figsize=(FIG_WIDTH * 1.6, FIG_HEIGHT * 1.5))

    none_df = df[df['security_standard_level'] == 'none']
    other_df = df[df['security_standard_level'] != 'none']

    if not none_df.empty:
        ax.plot(none_df['n'], none_df['actual_time'], marker='o', linewidth=1.2, markersize=3, label='n (Actual)')
        ax.plot(none_df['n'], none_df['estimated_time'], marker='s', linestyle='--', linewidth=1.2, markersize=3, label='n (Estimated)')
        
        for _, row in none_df.iterrows():
            percent_error = abs((row['estimated_time'] - row['actual_time']) / row['actual_time']) * 100
            mid_point = (row['actual_time'] + row['estimated_time']) / 2
            ax.vlines(row['n'], row['actual_time'], row['estimated_time'], 
                      colors='gray', linestyles='dotted', alpha=0.5, linewidth=0.8)
            
            # ax.text(row['n'], mid_point, f'{percent_error:.1f}%', 
            #         ha='center', va='center', fontsize=8)

    for security_level in other_df['security_standard_level'].unique():
        subset = df[df['security_standard_level'] == security_level]
        ax.plot(subset['n'], subset['actual_time'], marker='o', linewidth=1.2, markersize=3, label=f'{security_level} (Actual)')
        ax.plot(subset['n'], subset['estimated_time'], marker='s', linestyle='--', linewidth=1.2, markersize=3,
                 label=f'{security_level} (Estimated)')
        
        for _, row in subset.iterrows():
            percent_error = abs((row['estimated_time'] - row['actual_time']) / row['actual_time']) * 100
            mid_point = (row['actual_time'] + row['estimated_time']) / 2
            ax.vlines(row['n'], row['actual_time'], row['estimated_time'], 
                      colors='gray', linestyles='dotted', alpha=0.5, linewidth=0.8)

            # ax.text(row['n'], mid_point, f'{percent_error:.1f}%', 
            #         ha='center', va='center', fontsize=8)

    ax.set_xlabel('Ring Dimension (n)')
    ax.set_ylabel('Runtime (s)')
    ax.set_title(f'Estimated vs Actual Runtime ({config["name"]})', pad=3)
    ax.legend(
        loc='lower center',
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
        handlelength=1.2,
        labelspacing=0.2,
        columnspacing=0.8,
    )
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    
    ax.set_xticks(range(13, 18))
    
    comparison_plot_path = output_dir / f"{config['title']}_analysis.png"
    save_plot(fig, comparison_plot_path)
    print_status(f"Estimated vs actual runtime comparison saved to {comparison_plot_path}")
    return comparison_plot_path

def analyze_operation_contributions(df, config, output_dir):
    contribution_cols = [f'{op}_contribution' for op in config['operation_counts'].keys()]
    df['total_estimated'] = df[contribution_cols].sum(axis=1)
    
    for col in contribution_cols:
        df[f'{col}_percent'] = df[col] / df['total_estimated'] * 100
    
    fig, ax = plt.subplots(figsize=(FIG_WIDTH * 1.8, FIG_HEIGHT * 1.35))
    bottom = np.zeros(len(df))
    
    x_labels = []
    for _, row in df.iterrows():
        if row['security_standard_level'] == 'none':
            x_labels.append(f"n={row['n']}")
        else:
            x_labels.append(f"{row['security_standard_level']}\nn={row['n']}")
    
    for col in contribution_cols:
        ax.bar(
            range(len(df)),
            df[f'{col}_percent'],
            bottom=bottom,
            label=col.replace('_contribution', ''),
            width=0.7,
        )
        bottom += df[f'{col}_percent']
    
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(x_labels, rotation=25, ha='right')
    ax.set_ylabel('Contribution (%)')
    ax.set_title(f'Estimated Runtime Contribution Breakdown ({config["name"]})', pad=3)
    ax.set_ylim(0, 102)
    ax.grid(True, axis='y', alpha=0.3)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.legend(
        loc='lower center',
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
        handlelength=1.2,
        labelspacing=0.2,
        columnspacing=0.8,
    )
    operation_contributions_plot_path = output_dir / f"{config['title']}_operation_contributions.png"
    save_plot(fig, operation_contributions_plot_path)
    print_status(f"Operation type contributions saved to {operation_contributions_plot_path}")
    return operation_contributions_plot_path
    
def main():
    start_time = datetime.now()
    args = parse_arguments()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    apply_paper_plot_style()

    print_cryptoracle_banner()
    print_cli_summary(args)

    config = load_config(args.config)
    print_model_summary(config)
    
    print_status(f"Loading primitive results from {config['primitive_benchmarks']}")
    primitive_df = load_primitive_results(config['primitive_benchmarks'], config)

    print_status(f"\nCalculating estimated runtimes using {config['statistic']}")
    estimated_df = calculate_estimated_runtime(primitive_df, config['operation_counts'], config['statistic'])
    print_estimation_breakdown(estimated_df, config["operation_counts"])

    output_path = output_dir / f'estimated_{config["title"]}_{config["statistic"]}.csv'
    estimated_df.to_csv(output_path, index=False)
    print_status(f"Estimated runtimes saved to {output_path}")
    
    print_status("Generating plots")

    df = load_and_process_data(config['actual_benchmarks'], output_path, config)
    
    stats = calculate_statistics(df)
    print_section_header("Overall Statistical Analysis")
    print_reduced_statistics(stats)
    print()
    
    comparison_plot_path = plot_comparison(df, config, output_dir)
    operation_contributions_plot_path = analyze_operation_contributions(df, config, output_dir)
    
    none_df = df[df['security_standard_level'] == 'none']
    other_df = df[df['security_standard_level'] != 'none']
    
    if not none_df.empty:
        print_section_header("Analysis for n values (security_level = none)")
        for n_value in sorted(none_df['n'].unique()):
            subset = none_df[none_df['n'] == n_value]
            subset_stats = calculate_statistics(subset)
            print_argument("N:", str(int(n_value)))
            print_reduced_statistics(subset_stats, indent="  ")
            print()
    
    if not other_df.empty:
        print_section_header("Analysis by Security Level")
        for security_level in sorted(other_df['security_standard_level'].unique()):
            subset = other_df[other_df['security_standard_level'] == security_level]
            subset_stats = calculate_statistics(subset)
            print_argument("Security Level:", str(security_level))
            print_reduced_statistics(subset_stats, indent="  ")
            print()

    end_time = datetime.now()
    print_section_header("Summary")
    print_argument("Estimated CSV:", str(output_path))
    print_argument("Comparison Plot:", str(comparison_plot_path))
    print_argument("Contributions Plot:", str(operation_contributions_plot_path))
    print_argument("Samples Analyzed:", str(len(df)))
    print_argument("Mean Relative Error (%):", f"{stats['Mean Relative Error (%)']:.4f}")
    print()
    print_timestamp(f"Performance model completed in {(end_time - start_time).total_seconds():.3f} seconds.")
                    
if __name__ == "__main__":
    main()
