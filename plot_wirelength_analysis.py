import json
import os
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def extract_wirelength_data(results_folder):
    """
    Extract average wirelength data from all experiment result files
    
    Returns:
        dict: Nested dictionary structure: {depth: {algorithm: [wirelengths]}}
    """
    data = defaultdict(lambda: defaultdict(list))
    
    # List all JSON files in the results folder
    files = [f for f in os.listdir(results_folder) if f.endswith('comprehensive_results.json')]
    
    print(f"Found {len(files)} result files")
    
    for filename in files:
        # Parse filename to extract depth and run number
        # Format: depth_sweep_100qubits_depth{depth}_run{run}_test_{timestamp}_comprehensive_results.json
        try:
            parts = filename.split('_')
            # Find the part that starts with "depth" followed by numbers
            depth = None
            run_num = None
            
            for i, part in enumerate(parts):
                if part.startswith('depth') and len(part) > 5:
                    depth = int(part[5:])  # Remove 'depth' prefix
                elif part.startswith('run') and len(part) > 3:
                    run_num = int(part[3:])  # Remove 'run' prefix
            
            if depth is None:
                print(f"Could not parse depth from filename: {filename}")
                continue
            
            # Load JSON file
            filepath = os.path.join(results_folder, filename)
            with open(filepath, 'r') as f:
                result_data = json.load(f)
            
            # Extract wirelength data for each algorithm
            algorithm_results = result_data.get('algorithm_results', {})
            
            for algorithm in ['steiner_packing', 'steiner_pathfinder', 'steiner_tree']:
                if algorithm in algorithm_results:
                    metrics = algorithm_results[algorithm].get('metrics', {})
                    total_wirelength = metrics.get('total_wirelength')
                    
                    if total_wirelength is not None:
                        data[depth][algorithm].append(total_wirelength)
                        
        except (ValueError, IndexError, KeyError) as e:
            print(f"Error processing file {filename}: {e}")
            continue
    
    return dict(data)

def create_wirelength_plot(data, output_filename='wirelength_comparison.png'):
    """
    Create a bar plot showing total wirelength by depth and algorithm
    """
    # Prepare data for plotting
    depths = sorted(data.keys())
    algorithms = ['steiner_packing', 'steiner_pathfinder', 'steiner_tree']
    
    # Calculate averages and prepare data
    plot_data = []
    for depth in depths:
        row_data = {'Depth': depth}
        for algorithm in algorithms:
            wirelengths = data[depth].get(algorithm, [])
            if wirelengths:
                avg_wirelength = np.mean(wirelengths)
                std_wirelength = np.std(wirelengths)
                row_data[f'{algorithm}_avg'] = avg_wirelength
                row_data[f'{algorithm}_std'] = std_wirelength
                row_data[f'{algorithm}_count'] = len(wirelengths)
                print(f"Depth {depth}, {algorithm}: {avg_wirelength:.2f} ± {std_wirelength:.2f} (n={len(wirelengths)})")
            else:
                row_data[f'{algorithm}_avg'] = 0
                row_data[f'{algorithm}_std'] = 0
                row_data[f'{algorithm}_count'] = 0
                print(f"Depth {depth}, {algorithm}: No data")
        plot_data.append(row_data)
    
    df = pd.DataFrame(plot_data)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set up bar positions
    x = np.arange(len(depths))
    width = 0.25
    
    # Create bars for each algorithm
    bars1 = ax.bar(x - width, df['steiner_packing_avg'], width, 
                   yerr=df['steiner_packing_std'], label='Steiner Packing', 
                   alpha=0.8, capsize=5)
    bars2 = ax.bar(x, df['steiner_pathfinder_avg'], width, 
                   yerr=df['steiner_pathfinder_std'], label='Steiner Pathfinder', 
                   alpha=0.8, capsize=5)
    bars3 = ax.bar(x + width, df['steiner_tree_avg'], width, 
                   yerr=df['steiner_tree_std'], label='Steiner Tree', 
                   alpha=0.8, capsize=5)
    
    # Customize the plot
    ax.set_xlabel('Circuit Depth', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Wirelength', fontsize=12, fontweight='bold')
    ax.set_title('Total Wirelength Comparison Across Circuit Depths\n(100 Qubits, Multiple Algorithms)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(depths)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on top of bars
    def add_value_labels(bars, values, stds):
        for bar, val, std in zip(bars, values, stds):
            height = bar.get_height()
            if height > 0:  # Only add label if there's data
                ax.annotate(f'{val:.1f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height + std),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
    
    add_value_labels(bars1, df['steiner_packing_avg'], df['steiner_packing_std'])
    add_value_labels(bars2, df['steiner_pathfinder_avg'], df['steiner_pathfinder_std'])
    add_value_labels(bars3, df['steiner_tree_avg'], df['steiner_tree_std'])
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as '{output_filename}'")
    # plt.show()  # Comment this out for headless environments
    
    return df

def main():
    """Main function to run the analysis and create plots"""
    results_folder = 'routing_experiment_results'
    
    print("Extracting total wirelength data from experiment results...")
    data = extract_wirelength_data(results_folder)
    
    print("\nData summary:")
    for depth in sorted(data.keys()):
        print(f"Depth {depth}:")
        for algorithm in ['steiner_packing', 'steiner_pathfinder', 'steiner_tree']:
            count = len(data[depth].get(algorithm, []))
            print(f"  {algorithm}: {count} runs")
    
    print("\nCreating total wirelength comparison plot...")
    df_results = create_wirelength_plot(data)
    
    print(f"\nResults summary:")
    print(df_results[['Depth', 'steiner_packing_avg', 'steiner_pathfinder_avg', 'steiner_tree_avg']])
    
    # Save results to CSV
    df_results.to_csv('wirelength_results_summary.csv', index=False)
    print("\nResults saved to 'wirelength_results_summary.csv'")

if __name__ == "__main__":
    main()