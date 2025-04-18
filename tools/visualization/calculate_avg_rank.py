import numpy as np
import pandas as pd

# Define the method names from the table
methods = [
    "Auto²ML (def)", "Auto²ML (8B)", "Auto²ML (-ext)", 
    "Auto²ML (+env -ext)", "Auto²ML (-epi)", 
    "AIDE (def)", "AIDE (+ext)",
    "DS (def)", "AK (def)", "MLAB (def)"
]

# Dataset names from the table
datasets = [
    "abalone", "airbnb", "airlines", "bio", "camoseg", "cd18", "climate", 
    "covertype", "electric(H)", "flood", "fiqa", "gnad10", "ham10000", 
    "hateful", "isic2017", "funding", "memotion", "mldoc", "nn5(D)", 
    "petfinder", "roadseg", "rvlcdip", "solar(10m)", "clothing", "yolanda"
]

# Performance data from the table, using -9999 for missing values
performance_data = [
    # Auto²ML (def), Auto²ML (8B), Auto²ML (-ext), Auto²ML (+env -ext), Auto²ML (-epi), AIDE (def), AIDE (+ext), DS (def), AK (def), MLAB (def)
    [-2.13, -2.09, -2.22, -2.22, -2.13, -2.10, -2.19, -9999, -9999, -2.23],  # abalone
    [0.43, 0.41, 0.39, 0.40, 0.42, 0.39, 0.42, -9999, 0.25, -9999],  # airbnb
    [0.66, -9999, 0.64, 0.64, 0.66, -9999, -9999, -9999, -9999, -9999],  # airlines
    [0.80, 0.80, 0.88, 0.88, 0.80, 0.88, 0.88, 0.76, -9999, -9999],  # bio
    [0.83, -9999, -9999, -9999, 0.84, -9999, -9999, -9999, -9999, -9999],  # camoseg
    [0.45, -9999, -0.28, -0.63, 0.45, -9999, 0.24, -9999, -1.84, -9999],  # cd18
    [0.48, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999],  # climate
    [0.98, -9999, -9999, 0.89, 0.98, -9999, 0.74, 0.95, -9999, -9999],  # covertype
    [-1.44, -9999, -9999, -1.75, -1.41, -9999, -9999, -9999, -9999, -9999],  # electric(H)
    [0.68, -9999, 0.68, 0.58, 0.68, -9999, 0.70, -9999, -9999, -9999],  # flood
    [0.48, -9999, -9999, -9999, 0.38, -9999, -9999, -9999, -9999, -9999],  # fiqa
    [0.84, 0.84, 0.87, 0.88, 0.84, 0.91, -9999, -9999, -9999, -9999],  # gnad10
    [0.56, 0.57, -9999, 0.67, -9999, -9999, 0.78, -9999, -9999, -9999],  # ham10000
    [0.57, -9999, -9999, 0.38, 0.57, 0.47, 0.52, -9999, 0.34, -9999],  # hateful
    [0.75, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999],  # isic2017
    [0.44, 0.44, 0.43, 0.44, 0.44, -9999, 0.44, -9999, -9999, -9999],  # funding
    [0.50, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999],  # memotion
    [0.96, 0.95, 0.96, 0.96, 0.95, 0.96, -9999, -9999, -9999, -9999],  # mldoc
    [-0.76, -9999, -9999, -9999, -0.76, -9999, -9999, -9999, -9999, -9999],  # nn5(D)
    [0.39, -9999, 0.40, 0.36, 0.39, -9999, 0.38, 0.39, 0.39, -9999],  # petfinder
    [0.47, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999],  # roadseg
    [0.87, -9999, -9999, 0.89, 0.87, -9999, -9999, -9999, -9999, -9999],  # rvlcdip
    [-2.27, -9999, -9999, -9999, -1.29, -9999, -9999, -9999, -9999, -9999],  # solar(10m)
    [0.75, -9999, 0.61, 0.62, 0.75, -9999, 0.75, -9999, -9999, -9999],  # clothing
    [-8.53, -8.54, -8.92, -8.89, -8.53, -9999, -8.54, -9.60, -9999, -9.60]   # yolanda
]

def main():
    # Create a pandas DataFrame
    df = pd.DataFrame(performance_data, index=datasets, columns=methods)
    
    # Get ranks for each dataset (row)
    # We use ascending=False because higher values are better
    ranks_df = df.rank(axis=1, method='average', ascending=False)
    
    # Calculate average rank for each method (column)
    avg_ranks = ranks_df.mean()
    
    # Calculate success rate for each method
    success_mask = (df != -9999)
    success_rate = success_mask.mean() * 100
    
    # Print results
    print("Average Ranks (lower is better):")
    for method, avg_rank in avg_ranks.items():
        print(f"{method}: {avg_rank:.2f} (Success rate: {success_rate[method]:.1f}%)")
    
    # Create a results DataFrame for better visualization
    results_df = pd.DataFrame({
        'Method': methods,
        'Avg Rank': avg_ranks.values,
        'Success Rate (%)': success_rate.values
    })
    results_df = results_df.sort_values('Avg Rank')
    
    print("\nSorted by Average Rank (best to worst):")
    print(results_df.to_string(index=False))
    
    # Display the number of datasets
    print(f"\nTotal number of datasets: {len(datasets)}")
    
    # Print the ranks for each dataset for verification
    print("\nRanks for each dataset (1 is best):")
    print(ranks_df)

if __name__ == "__main__":
    main()