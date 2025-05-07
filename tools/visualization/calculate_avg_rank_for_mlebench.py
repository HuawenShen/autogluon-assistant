import numpy as np
import pandas as pd

# Define the method names from the table
methods = ["Ours", "AIDE", "MLAB", "OD"]

# Dataset names from the table (D1-D21)
datasets = [f"D{i}" for i in range(1, 22)]

# Performance data from the table, using -9999 for missing values (X)
performance_data = [
    # Ours, AIDE, MLAB, OD
    [1.000, 1.000, 0.943, 0.495],      # D1
    [0.904, 0.855, 0.712, -9999],      # D2
    [-9999, -9999, -9999, -0.220],     # D3
    [0.936, -9999, 0.852, 0.884],      # D4
    [-0.442, -0.694, -4.800, -9999],   # D5
    [-0.008, -0.817, -12.759, -0.426], # D6
    [0.998, 0.996, -9999, 0.853],      # D7
    [-9999, 0.903, 0.953, 0.971],      # D8
    [-0.242, -0.801, -9999, -0.934],   # D9
    [-9999, -9999, -9999, -9999],      # D10
    [-5.111, -5.463, -10.022, -1053.080], # D11
    [-0.059, -0.069, -0.063, -0.542],  # D12
    [0.990, 0.962, 0.817, 0.494],      # D13
    [0.787, 0.642, 0.500, 0.684],      # D14
    [0.673, 0.859, 0.421, 0.635],      # D15
    [-0.384, -0.426, -0.555, -0.563],  # D16
    [0.963, 0.958, 0.943, 0.958],      # D17
    [0.960, 0.899, -9999, -9999],      # D18
    [-9999, 0.991, -9999, -9999],      # D19
    [0.958, -9999, -9999, -9999],      # D20
    [0.625, 0.869, -9999, 0.914]       # D21
]

def main():
    # Create a pandas DataFrame (transposed from original to match row=datasets, col=methods)
    df = pd.DataFrame(performance_data, index=datasets, columns=methods)
    
    # Get ranks for each dataset (row)
    # We use ascending=False because higher values are better
    ranks_df = df.rank(axis=1, method='average', ascending=False)
    
    # Calculate average rank for each method (column)
    avg_ranks = ranks_df.mean()
    
    # Calculate success rate for each method (percentage of non-missing values)
    success_mask = (df != -9999)
    success_rate = success_mask.mean() * 100
    
    # Count number of best performances (rank 1) for each method
    best_performances = (ranks_df == 1).sum()
    
    # Count number of top-3 placements for each method
    top3_performances = (ranks_df <= 3).sum()
    
    # Print results
    print("Average Ranks (lower is better):")
    for method, avg_rank in avg_ranks.items():
        print(f"{method}: {avg_rank:.2f} (Success rate: {success_rate[method]:.1f}%)")
    
    # Create a results DataFrame for better visualization
    results_df = pd.DataFrame({
        'Method': methods,
        'Avg Rank': avg_ranks.values,
        'Success Rate (%)': success_rate.values,
        'Best Performances': best_performances.values,
        'Top-3 Placements': top3_performances.values
    })
    results_df = results_df.sort_values('Avg Rank')
    
    print("\nSorted by Average Rank (best to worst):")
    print(results_df.to_string(index=False))
    
    # Display the number of datasets
    print(f"\nTotal number of datasets: {len(datasets)}")
    
    # Print the ranks for each dataset for verification
    print("\nRanks for each dataset (1 is best):")
    print(ranks_df)
    
    # Additional metrics from the table
    # Count how many times each method is underlined (above median)
    # For simplicity, we'll calculate this directly
    median_values = df.median(axis=1)
    above_median = df.ge(median_values, axis=0)
    above_median_count = above_median.sum()
    
    print("\nNumber of times each method performs above median:")
    for method, count in above_median_count.items():
        print(f"{method}: {count}")
    
    # Gold, silver, bronze counts (top 3 performers)
    gold_counts = (ranks_df == 1).sum()
    silver_counts = (ranks_df == 2).sum()
    bronze_counts = (ranks_df == 3).sum()
    
    print("\nMedal counts:")
    medal_df = pd.DataFrame({
        'Method': methods,
        'Gold': gold_counts.values,
        'Silver': silver_counts.values,
        'Bronze': bronze_counts.values,
        'Total Medals': gold_counts.values + silver_counts.values + bronze_counts.values
    })
    print(medal_df.to_string(index=False))

if __name__ == "__main__":
    main()