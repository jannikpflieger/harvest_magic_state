"""
Results analysis utility for the comprehensive routing pipeline.
Processes JSON results files and generates statistical analysis, comparisons, and reports.
"""

import os
import json
import statistics
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


class RoutingResultsAnalyzer:
    """Analyzer for routing pipeline results."""
    
    def __init__(self, results_dir: str):
        """Initialize analyzer with results directory."""
        self.results_dir = Path(results_dir)
        self.results_data = []
        self.batch_data = []
        
    def load_results(self) -> int:
        """Load all JSON result files from the results directory."""
        loaded_count = 0
        
        if not self.results_dir.exists():
            print(f"Results directory {self.results_dir} does not exist.")
            return 0
        
        # Load individual test results
        for json_file in self.results_dir.glob("*_comprehensive_results.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    data['_source_file'] = str(json_file)
                    self.results_data.append(data)
                    loaded_count += 1
            except Exception as e:
                print(f"Failed to load {json_file}: {e}")
        
        # Load batch results
        for json_file in self.results_dir.glob("*_batch_results.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    data['_source_file'] = str(json_file)
                    self.batch_data.append(data)
                    loaded_count += 1
            except Exception as e:
                print(f"Failed to load batch file {json_file}: {e}")
        
        print(f"Loaded {len(self.results_data)} individual results and {len(self.batch_data)} batch results")
        return loaded_count
    
    def generate_summary_report(self) -> str:
        """Generate a comprehensive summary report."""
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("ROUTING PIPELINE RESULTS SUMMARY REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Results Directory: {self.results_dir}")
        report_lines.append("")
        
        # Overall statistics
        report_lines.append("OVERALL STATISTICS")
        report_lines.append("-" * 40)
        report_lines.append(f"Total individual tests: {len(self.results_data)}")
        report_lines.append(f"Total batch tests: {len(self.batch_data)}")
        
        # Algorithm success rates
        algorithm_stats = self._calculate_algorithm_statistics()
        report_lines.append("")
        report_lines.append("ALGORITHM SUCCESS RATES")
        report_lines.append("-" * 40)
        
        for alg_name, stats in algorithm_stats.items():
            success_rate = stats['successes'] / max(stats['total_tests'], 1) * 100
            report_lines.append(f"{alg_name}: {stats['successes']}/{stats['total_tests']} ({success_rate:.1f}%)")
        
        # Performance metrics
        performance_stats = self._calculate_performance_statistics()
        report_lines.append("")
        report_lines.append("PERFORMANCE METRICS (Successful Tests Only)")
        report_lines.append("-" * 40)
        
        for metric_name, metric_data in performance_stats.items():
            if metric_data['values']:
                mean_val = statistics.mean(metric_data['values'])
                median_val = statistics.median(metric_data['values'])
                min_val = min(metric_data['values'])
                max_val = max(metric_data['values'])
                
                report_lines.append(f"{metric_name}:")
                report_lines.append(f"  Mean: {mean_val:.2f}, Median: {median_val:.2f}")
                report_lines.append(f"  Range: {min_val:.2f} - {max_val:.2f}")
        
        # Layout analysis
        layout_stats = self._analyze_layout_performance()
        report_lines.append("")
        report_lines.append("LAYOUT SIZE ANALYSIS")
        report_lines.append("-" * 40)
        
        for layout_key, layout_data in layout_stats.items():
            success_rate = layout_data['successful_tests'] / max(layout_data['total_tests'], 1) * 100
            avg_runtime = statistics.mean(layout_data['runtimes']) if layout_data['runtimes'] else 0
            
            report_lines.append(f"{layout_key}: {layout_data['successful_tests']}/{layout_data['total_tests']} "
                              f"({success_rate:.1f}%), avg runtime: {avg_runtime:.1f}ms")
        
        # Circuit source analysis
        circuit_stats = self._analyze_circuit_sources()
        if circuit_stats:
            report_lines.append("")
            report_lines.append("CIRCUIT SOURCE ANALYSIS")
            report_lines.append("-" * 40)
            
            for source, source_data in circuit_stats.items():
                success_rate = source_data['successful_tests'] / max(source_data['total_tests'], 1) * 100
                report_lines.append(f"{source}: {source_data['successful_tests']}/{source_data['total_tests']} ({success_rate:.1f}%)")
        
        report_text = "\n".join(report_lines)
        
        # Save report to file
        report_file = self.results_dir / f"summary_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        try:
            with open(report_file, 'w') as f:
                f.write(report_text)
            print(f"Summary report saved to: {report_file}")
        except Exception as e:
            print(f"Failed to save report: {e}")
        
        return report_text
    
    def _calculate_algorithm_statistics(self) -> Dict:
        """Calculate success rates and statistics for each algorithm."""
        algorithm_stats = {}
        
        for result in self.results_data:
            if 'algorithm_results' in result:
                for alg_name, alg_data in result['algorithm_results'].items():
                    if alg_name not in algorithm_stats:
                        algorithm_stats[alg_name] = {
                            'total_tests': 0,
                            'successes': 0,
                            'failures': 0,
                            'runtimes': [],
                            'error_types': {}
                        }
                    
                    algorithm_stats[alg_name]['total_tests'] += 1
                    
                    if alg_data.get('status') == 'SUCCESS':
                        algorithm_stats[alg_name]['successes'] += 1
                        if 'runtime_ms' in alg_data:
                            algorithm_stats[alg_name]['runtimes'].append(alg_data['runtime_ms'])
                    else:
                        algorithm_stats[alg_name]['failures'] += 1
                        error = alg_data.get('error', 'Unknown error')
                        algorithm_stats[alg_name]['error_types'][error] = algorithm_stats[alg_name]['error_types'].get(error, 0) + 1
        
        return algorithm_stats
    
    def _calculate_performance_statistics(self) -> Dict:
        """Calculate performance statistics across all successful tests."""
        metrics_to_analyze = [
            'total_runtime_ms',
            'avg_nets_packed_per_timestep',
            'total_wirelength',
            'avg_wirelength_per_net',
            'fraction_nets_packed_first_try',
            'avg_utilization',
            'peak_utilization'
        ]
        
        performance_stats = {}
        
        for metric in metrics_to_analyze:
            performance_stats[metric] = {
                'values': [],
                'by_algorithm': {}
            }
        
        for result in self.results_data:
            # Test-level metrics
            test_runtime = result.get('test_metadata', {}).get('total_runtime_ms', 0)
            if test_runtime > 0:
                performance_stats['total_runtime_ms']['values'].append(test_runtime)
            
            # Algorithm-level metrics
            if 'algorithm_results' in result:
                for alg_name, alg_data in result['algorithm_results'].items():
                    if alg_data.get('status') == 'SUCCESS':
                        metrics = alg_data.get('metrics', {})
                        
                        for metric in metrics_to_analyze[1:]:  # Skip runtime as it's handled above
                            if metric in metrics:
                                value = metrics[metric]
                                performance_stats[metric]['values'].append(value)
                                
                                # Track by algorithm
                                if alg_name not in performance_stats[metric]['by_algorithm']:
                                    performance_stats[metric]['by_algorithm'][alg_name] = []
                                performance_stats[metric]['by_algorithm'][alg_name].append(value)
        
        return performance_stats
    
    def _analyze_layout_performance(self) -> Dict:
        """Analyze performance by layout size."""
        layout_stats = {}
        
        for result in self.results_data:
            layout_info = result.get('layout_info', {})
            if layout_info:
                rows = layout_info.get('rows', 0)
                cols = layout_info.get('cols', 0)
                layout_key = f"{rows}x{cols}"
                
                if layout_key not in layout_stats:
                    layout_stats[layout_key] = {
                        'total_tests': 0,
                        'successful_tests': 0,
                        'runtimes': [],
                        'total_patches': rows * cols
                    }
                
                layout_stats[layout_key]['total_tests'] += 1
                
                # Check if test was successful (has successful algorithms)
                successful_algs = result.get('comparative_analysis', {}).get('successful_algorithms', [])
                if successful_algs:
                    layout_stats[layout_key]['successful_tests'] += 1
                
                runtime = result.get('test_metadata', {}).get('total_runtime_ms', 0)
                if runtime > 0:
                    layout_stats[layout_key]['runtimes'].append(runtime)
        
        return layout_stats
    
    def _analyze_circuit_sources(self) -> Dict:
        """Analyze performance by circuit source type."""
        source_stats = {}
        
        for result in self.results_data:
            circuit_info = result.get('circuit_info', {})
            source = circuit_info.get('source', 'unknown')
            
            if source not in source_stats:
                source_stats[source] = {
                    'total_tests': 0,
                    'successful_tests': 0,
                    'runtimes': []
                }
            
            source_stats[source]['total_tests'] += 1
            
            # Check if test was successful
            successful_algs = result.get('comparative_analysis', {}).get('successful_algorithms', [])
            if successful_algs:
                source_stats[source]['successful_tests'] += 1
            
            runtime = result.get('test_metadata', {}).get('total_runtime_ms', 0)
            if runtime > 0:
                source_stats[source]['runtimes'].append(runtime)
        
        return source_stats
    
    def create_comparison_dataframe(self) -> pd.DataFrame:
        """Create a pandas DataFrame with all results for detailed analysis."""
        rows = []
        
        for result in self.results_data:
            base_info = {
                'test_id': result.get('test_metadata', {}).get('test_id', 'unknown'),
                'timestamp': result.get('test_metadata', {}).get('timestamp', ''),
                'circuit_source': result.get('circuit_info', {}).get('source', 'unknown'),
                'layout_rows': result.get('layout_info', {}).get('rows', 0),
                'layout_cols': result.get('layout_info', {}).get('cols', 0),
                'layout_patches': result.get('layout_info', {}).get('total_patches', 0),
                'test_runtime_ms': result.get('test_metadata', {}).get('total_runtime_ms', 0),
                'num_dag_operations': result.get('test_metadata', {}).get('num_dag_operations', 0)
            }
            
            # Add circuit info if available
            circuit_info = result.get('circuit_info', {})
            if isinstance(circuit_info, dict):
                base_info.update({
                    'circuit_num_qubits': circuit_info.get('num_qubits', 0),
                    'circuit_depth': circuit_info.get('depth', 0),
                    'circuit_total_gates': circuit_info.get('total_gates', 0),
                    'pcb_successful': circuit_info.get('pcb_conversion_successful', False),
                    'dag_successful': circuit_info.get('dag_analysis_successful', False)
                })
            
            # Create one row per algorithm
            algorithm_results = result.get('algorithm_results', {})
            if algorithm_results:
                for alg_name, alg_data in algorithm_results.items():
                    row = base_info.copy()
                    row.update({
                        'algorithm': alg_name,
                        'algorithm_status': alg_data.get('status', 'UNKNOWN'),
                        'algorithm_runtime_ms': alg_data.get('runtime_ms', 0),
                        'algorithm_success': alg_data.get('status') == 'SUCCESS'
                    })
                    
                    # Add algorithm metrics
                    metrics = alg_data.get('metrics', {})
                    for metric_name, metric_value in metrics.items():
                        row[f'metric_{metric_name}'] = metric_value
                    
                    rows.append(row)
            else:
                # If no algorithm results, add base info only
                rows.append(base_info)
        
        df = pd.DataFrame(rows)
        
        # Save DataFrame to CSV
        csv_file = self.results_dir / f"comprehensive_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        try:
            df.to_csv(csv_file, index=False)
            print(f"Results DataFrame saved to: {csv_file}")
        except Exception as e:
            print(f"Failed to save CSV: {e}")
        
        return df
    
    def create_performance_plots(self, save_plots: bool = True) -> Dict:
        """Create performance visualization plots."""
        plots_created = {}
        
        if not self.results_data:
            print("No data available for plotting.")
            return plots_created
        
        try:
            # Create comparison DataFrame
            df = self.create_comparison_dataframe()
            
            if df.empty:
                print("DataFrame is empty, cannot create plots.")
                return plots_created
            
            # Filter for successful algorithm runs only
            successful_df = df[df['algorithm_success'] == True].copy()
            
            if successful_df.empty:
                print("No successful algorithm runs found for plotting.")
                return plots_created
            
            # Set up plotting style
            plt.style.use('default')
            sns.set_palette("husl")
            
            # Plot 1: Algorithm runtime comparison
            if 'algorithm_runtime_ms' in successful_df.columns and 'algorithm' in successful_df.columns:
                plt.figure(figsize=(10, 6))
                sns.boxplot(data=successful_df, x='algorithm', y='algorithm_runtime_ms')
                plt.title('Algorithm Runtime Comparison')
                plt.ylabel('Runtime (ms)')
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                if save_plots:
                    plot_file = self.results_dir / 'algorithm_runtime_comparison.png'
                    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
                    plots_created['runtime_comparison'] = str(plot_file)
                
                plt.close()
            
            # Plot 2: Layout size vs performance
            if 'layout_patches' in successful_df.columns and 'algorithm_runtime_ms' in successful_df.columns:
                plt.figure(figsize=(10, 6))
                sns.scatterplot(data=successful_df, x='layout_patches', y='algorithm_runtime_ms', 
                               hue='algorithm', style='algorithm', s=100)
                plt.title('Runtime vs Layout Size')
                plt.xlabel('Number of Layout Patches')
                plt.ylabel('Algorithm Runtime (ms)')
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()
                
                if save_plots:
                    plot_file = self.results_dir / 'runtime_vs_layout_size.png'
                    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
                    plots_created['layout_scaling'] = str(plot_file)
                
                plt.close()
            
            # Plot 3: Algorithm success rates
            algorithm_stats = self._calculate_algorithm_statistics()
            if algorithm_stats:
                algorithms = list(algorithm_stats.keys())
                success_rates = [stats['successes'] / max(stats['total_tests'], 1) * 100 
                               for stats in algorithm_stats.values()]
                
                plt.figure(figsize=(8, 6))
                bars = plt.bar(algorithms, success_rates, color=['skyblue', 'lightcoral', 'lightgreen'])
                plt.title('Algorithm Success Rates')
                plt.ylabel('Success Rate (%)')
                plt.ylim(0, 100)
                
                # Add value labels on bars
                for i, (bar, rate) in enumerate(zip(bars, success_rates)):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                            f'{rate:.1f}%', ha='center', va='bottom')
                
                plt.tight_layout()
                
                if save_plots:
                    plot_file = self.results_dir / 'algorithm_success_rates.png'
                    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
                    plots_created['success_rates'] = str(plot_file)
                
                plt.close()
            
            print(f"Created {len(plots_created)} performance plots")
            
        except Exception as e:
            print(f"Error creating plots: {e}")
        
        return plots_created
    
    def print_detailed_analysis(self):
        """Print detailed analysis to console."""
        print(self.generate_summary_report())
        
        if self.results_data:
            print("\n" + "=" * 80)
            print("DETAILED BREAKDOWN")
            print("=" * 80)
            
            # Most recent tests
            recent_tests = sorted(self.results_data, 
                                key=lambda x: x.get('test_metadata', {}).get('timestamp', ''), 
                                reverse=True)[:5]
            
            print(f"\nMOST RECENT TESTS ({len(recent_tests)}):")
            print("-" * 40)
            
            for result in recent_tests:
                test_id = result.get('test_metadata', {}).get('test_id', 'unknown')
                timestamp = result.get('test_metadata', {}).get('timestamp', '')
                runtime = result.get('test_metadata', {}).get('total_runtime_ms', 0)
                
                successful_algs = result.get('comparative_analysis', {}).get('successful_algorithms', [])
                failed_algs = result.get('comparative_analysis', {}).get('failed_algorithms', [])
                
                print(f"  {test_id} ({timestamp[:19]})")
                print(f"    Runtime: {runtime:.1f}ms")
                print(f"    Successful: {successful_algs}")
                if failed_algs:
                    print(f"    Failed: {failed_algs}")
                print()


def main():
    """Main function for running analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze routing pipeline results")
    parser.add_argument("results_dir", help="Directory containing results JSON files")
    parser.add_argument("--plots", action="store_true", help="Create performance plots")
    parser.add_argument("--detailed", action="store_true", help="Show detailed analysis")
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = RoutingResultsAnalyzer(args.results_dir)
    
    # Load results
    loaded_count = analyzer.load_results()
    
    if loaded_count == 0:
        print("No results found to analyze.")
        return
    
    print(f"Loaded {loaded_count} result files from {args.results_dir}")
    
    # Generate summary report
    print("\nGenerating summary report...")
    analyzer.generate_summary_report()
    
    # Create DataFrame for detailed analysis
    print("\nCreating comparison DataFrame...")
    df = analyzer.create_comparison_dataframe()
    print(f"DataFrame created with {len(df)} rows and {len(df.columns)} columns")
    
    # Create plots if requested
    if args.plots:
        print("\nCreating performance plots...")
        plots = analyzer.create_performance_plots()
        for plot_name, plot_file in plots.items():
            print(f"  Created {plot_name}: {plot_file}")
    
    # Show detailed analysis if requested
    if args.detailed:
        analyzer.print_detailed_analysis()
    
    print("\nAnalysis completed!")


if __name__ == "__main__":
    # If run without arguments, analyze current directory
    if len(os.sys.argv) == 1:
        current_dir = Path.cwd()
        results_dirs = [d for d in current_dir.iterdir() if d.is_dir() and 'results' in d.name.lower()]
        
        if results_dirs:
            print(f"Found {len(results_dirs)} potential results directories:")
            for i, d in enumerate(results_dirs):
                print(f"  {i+1}. {d}")
            
            try:
                choice = int(input(f"Select directory (1-{len(results_dirs)}): ")) - 1
                if 0 <= choice < len(results_dirs):
                    analyzer = RoutingResultsAnalyzer(str(results_dirs[choice]))
                    analyzer.load_results()
                    analyzer.print_detailed_analysis()
                    analyzer.create_performance_plots()
                else:
                    print("Invalid choice.")
            except (ValueError, KeyboardInterrupt):
                print("Analysis cancelled.")
        else:
            print("No results directories found in current directory.")
            print("Usage: python routing_results_analyzer.py <results_directory>")
    else:
        main()