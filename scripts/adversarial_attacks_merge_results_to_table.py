import argparse
import os
import pandas as pd
import re


def parse_subdir_name(subdir_name):
    """
    Parse subdirectory name to extract model_name, optimizer_name, eps, and alpha.
    
    Expected format from format_optimizer_subdir:
    {model_name}_{optimizer_name}_eps_{eps}_alpha_{alpha}_{other_params}
    
    Example: cifar10_de_eps_0.1_alpha_1.0_pop_size_500_eps_0.1_wf_0.8_cr_0.9_strategy_0
    """
    parts = subdir_name.split('_')
    
    if len(parts) < 4:
        return None
    
    # Extract model_name (first part)
    model_name = parts[0]
    
    # Extract optimizer_name (second part)
    optimizer_name = parts[1]
    
    # Extract eps (look for eps_VALUE pattern, first occurrence after model and optimizer)
    eps = None
    alpha = None
    
    i = 2
    while i < len(parts):
        if parts[i] == 'eps' and i + 1 < len(parts):
            if eps is None:  # Take first occurrence
                try:
                    eps = float(parts[i + 1])
                except ValueError:
                    pass
            i += 2
        elif parts[i] == 'alpha' and i + 1 < len(parts):
            try:
                alpha = float(parts[i + 1])
            except ValueError:
                pass
            i += 2
        else:
            i += 1
    
    if eps is None or alpha is None:
        return None
    
    return {
        'model_name': model_name,
        'optimizer_name': optimizer_name,
        'eps': eps,
        'alpha': alpha
    }


def merge_results(output_dir, output_file='merged_results.csv'):
    """
    Merge aggregation.csv files from all optimizer subdirectories.
    
    Args:
        output_dir: Directory containing optimizer subdirectories
        output_file: Output CSV file name
    """
    results = []
    
    # Iterate through subdirectories
    for subdir in os.listdir(output_dir):
        subdir_path = os.path.join(output_dir, subdir)
        
        # Check if it's a directory
        if not os.path.isdir(subdir_path):
            continue
        
        # Check if aggregation.csv exists
        aggregation_file = os.path.join(subdir_path, 'aggregation.csv')
        if not os.path.exists(aggregation_file):
            print(f"Warning: No aggregation.csv found in {subdir}")
            continue
        
        # Parse subdirectory name
        parsed = parse_subdir_name(subdir)
        if parsed is None:
            print(f"Warning: Could not parse subdirectory name: {subdir}")
            continue
        
        # Read aggregation.csv
        try:
            df = pd.read_csv(aggregation_file)
            
            # Should contain only 1 row of data
            if len(df) == 0:
                print(f"Warning: Empty aggregation.csv in {subdir}")
                continue
            
            # Take the first row (should be only one)
            row = df.iloc[0].to_dict()
            
            # Add parsed metadata
            row['model_name'] = parsed['model_name']
            row['optimizer_name'] = parsed['optimizer_name']
            row['eps'] = parsed['eps']
            row['alpha'] = parsed['alpha']
            
            results.append(row)
            print(f"Processed: {subdir}")
            
        except Exception as e:
            print(f"Error reading {aggregation_file}: {e}")
            continue
    
    if not results:
        print("No results to merge!")
        return
    
    # Create DataFrame with metadata columns first
    df_results = pd.DataFrame(results)
    
    # Reorder columns: metadata first, then metrics
    metadata_cols = ['model_name', 'optimizer_name', 'eps', 'alpha']
    metric_cols = [col for col in df_results.columns if col not in metadata_cols]
    df_results = df_results[metadata_cols + metric_cols]
    
    # Sort by model_name, eps, alpha, optimizer_name
    df_results = df_results.sort_values(by=['model_name', 'eps', 'alpha', 'optimizer_name'])
    
    # Save to file
    df_results.to_csv(output_file, index=False)
    print(f"\nMerged results saved to: {output_file}")
    print(f"Total experiments: {len(df_results)}")
    print(f"\nColumns: {list(df_results.columns)}")
    print(f"\nPreview:")
    print(df_results.head())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Merge aggregation.csv files from optimizer subdirectories into a single table'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directory containing optimizer subdirectories (e.g., output/)'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default='merged_results.csv',
        help='Output CSV filename (default: merged_results.csv)'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        print(f"Error: Directory {args.output_dir} does not exist")
        exit(1)
    
    merge_results(args.output_dir, args.output_file)
