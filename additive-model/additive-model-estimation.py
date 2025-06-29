import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import sys
import argparse
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.append(str(Path(__file__).parent.parent))
from src.logging_utils import print_status, print_error

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

def load_config(config_path):
    """Load and validate configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        required_fields = ['name', 'primitive_benchmarks', 'actual_benchmarks', 
                         'statistic', 'title', 'operation_counts']
        
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field '{field}' in configuration")
        
        if config['statistic'] not in ['mean', 'median']:
            raise ValueError("statistic must be either 'mean' or 'median'")
        
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
    
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
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
    
    plt.tight_layout()
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    security_plot_path = output_dir / f"estimated_{config['title']}_security_{config['statistic']}.png"
    fig1.savefig(security_plot_path)
    print_status(f"Security standards comparison saved to {security_plot_path}")
    
    fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(12, 10))
 
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
    
    plt.tight_layout()
    
    n_plot_path = output_dir / f"estimated_{config['title']}_n_{config['statistic']}.png"
    fig2.savefig(n_plot_path)
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

def plot_comparison(df, config, output_dir):
    df = df[df['n'] != 12]

    plt.figure(figsize=(12, 8))

    none_df = df[df['security_standard_level'] == 'none']
    other_df = df[df['security_standard_level'] != 'none']

    if not none_df.empty:
        plt.plot(none_df['n'], none_df['actual_time'], marker='o', label='n (Actual)')
        plt.plot(none_df['n'], none_df['estimated_time'], marker='s', linestyle='--', label='n (Estimated)')
        
        for _, row in none_df.iterrows():
            percent_error = abs((row['estimated_time'] - row['actual_time']) / row['actual_time']) * 100
            mid_point = (row['actual_time'] + row['estimated_time']) / 2
            plt.vlines(row['n'], row['actual_time'], row['estimated_time'], 
                      colors='gray', linestyles='dotted', alpha=0.5)
            
            plt.text(row['n'], mid_point, f'{percent_error:.1f}%', 
                    ha='center', va='center', fontsize=8)

    for security_level in other_df['security_standard_level'].unique():
        subset = df[df['security_standard_level'] == security_level]
        plt.plot(subset['n'], subset['actual_time'], marker='o', label=f'{security_level} (Actual)')
        plt.plot(subset['n'], subset['estimated_time'], marker='s', linestyle='--', 
                 label=f'{security_level} (Estimated)')
        
        for _, row in subset.iterrows():
            percent_error = abs((row['estimated_time'] - row['actual_time']) / row['actual_time']) * 100
            mid_point = (row['actual_time'] + row['estimated_time']) / 2
            plt.vlines(row['n'], row['actual_time'], row['estimated_time'], 
                      colors='gray', linestyles='dotted', alpha=0.5)

            plt.text(row['n'], mid_point, f'{percent_error:.1f}%', 
                    ha='center', va='center', fontsize=8)

    plt.xlabel('n')
    plt.ylabel('Time (s)')
    plt.title(f'Estimated vs Actual Runtime for values of n ({config["name"]})')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.xticks(range(13, 18))
    
    plt.tight_layout()
    comparison_plot_path = output_dir + f"/{config['title']}_analysis.png"
    plt.savefig(comparison_plot_path)
    print_status(f"Estimated vs actual runtime comparison saved to {comparison_plot_path}")

def analyze_operation_contributions(df, config, output_dir):
    contribution_cols = [f'{op}_contribution' for op in config['operation_counts'].keys()]
    df['total_estimated'] = df[contribution_cols].sum(axis=1)
    
    for col in contribution_cols:
        df[f'{col}_percent'] = df[col] / df['total_estimated'] * 100
    
    plt.figure(figsize=(12, 6))
    bottom = np.zeros(len(df))
    
    x_labels = []
    for _, row in df.iterrows():
        if row['security_standard_level'] == 'none':
            x_labels.append(f"n={row['n']}")
        else:
            x_labels.append(f"{row['security_standard_level']}\nn={row['n']}")
    
    for col in contribution_cols:
        plt.bar(range(len(df)), df[f'{col}_percent'], bottom=bottom, 
                label=col.replace('_contribution', ''))
        bottom += df[f'{col}_percent']
    
    plt.xticks(range(len(df)), x_labels, rotation=45)
    plt.ylabel('Contribution (%)')
    plt.title(f'Operation Type Contributions to Estimated Runtime ({config["name"]})')
    plt.legend()
    plt.tight_layout()
    operation_contributions_plot_path = output_dir + f"{config['title']}_operation_contributions.png"
    plt.savefig(operation_contributions_plot_path)
    print_status(f"Operation type contributions saved to {operation_contributions_plot_path}")
    
def main():
    args = parse_arguments()
    config = load_config(args.config)
    
    print_status(f"Loading primitive results from {config['primitive_benchmarks']}")
    primitive_df = load_primitive_results(config['primitive_benchmarks'], config)

    print_status(f"\nCalculating estimated runtimes using {config['statistic']}")
    estimated_df = calculate_estimated_runtime(primitive_df, config['operation_counts'], config['statistic'])

    print_status("\nNumber of data points used for each calculation:")
    for _, row in estimated_df.iterrows():
        sec_level = row['security_standard_level']
        n_value = row['n']
        print(f"\nSecurity Level: {sec_level}, n: {n_value}")
        for operation in config['operation_counts'].keys():
            count_col = f'{operation}_count'
            if count_col in row:
                print(f"  {operation}: {row[count_col]} data points")
        print(f"  Estimated Runtime: {row['estimated_time']:.3f} s")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'estimated_{config["title"]}_{config["statistic"]}.csv'
    estimated_df.to_csv(output_path, index=False)
    print_status(f"Estimated runtimes saved to {output_path}")
    
    print_status("Generating plots")
    plot_estimated_runtimes(estimated_df, config, args.output_dir)

    df = load_and_process_data(config['actual_benchmarks'], output_path, config)
    
    stats = calculate_statistics(df)
    print("\nOverall Statistical Analysis:")
    for metric, value in stats.items():
        if metric == 'Number of Samples':
            print(f"{metric}: {int(value)}")
        elif pd.isna(value):
            print(f"{metric}: undefined (needs more samples)")
        else:
            print(f"{metric}: {value:.4f}")
    
    plot_comparison(df, config, args.output_dir)
    analyze_operation_contributions(df, config, args.output_dir)
    
    none_df = df[df['security_standard_level'] == 'none']
    other_df = df[df['security_standard_level'] != 'none']
    
    if not none_df.empty:
        print("\nAnalysis for n values (security_level = none):")
        for n_value in sorted(none_df['n'].unique()):
            subset = none_df[none_df['n'] == n_value]
            subset_stats = calculate_statistics(subset)
            print(f"\nn = {n_value}:")
            for metric, value in subset_stats.items():
                if metric == 'Number of Samples':
                    print(f"  {metric}: {int(value)}")
                elif pd.isna(value):
                    print(f"  {metric}: undefined (needs more samples)")
                else:
                    print(f"  {metric}: {value:.4f}")
    
    if not other_df.empty:
        print("\nAnalysis by Security Level:")
        for security_level in sorted(other_df['security_standard_level'].unique()):
            subset = other_df[other_df['security_standard_level'] == security_level]
            subset_stats = calculate_statistics(subset)
            print(f"\n{security_level}:")
            for metric, value in subset_stats.items():
                if metric == 'Number of Samples':
                    print(f"  {metric}: {int(value)}")
                elif pd.isna(value):
                    print(f"  {metric}: undefined (needs more samples)")
                else:
                    print(f"  {metric}: {value:.4f}")
                    
if __name__ == "__main__":
    main()