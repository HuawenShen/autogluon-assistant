import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.weight'] = 'bold'  # Set all fonts to bold by default

# Define the data
retrieval_counts = [0, 1, 3, 5, 10]
tasks = 8

# Success rates (1 or 0)
success_data = np.array([
    [1, 1, 1, 0, 1, 0, 0, 1],  # Retrieved 0
    [1, 1, 0, 1, 1, 0, 0, 1],  # Retrieved 1
    [1, 1, 1, 1, 1, 0, 1, 1],  # Retrieved 3
    [1, 1, 1, 1, 1, 1, 1, 1],  # Retrieved 5 (baseline)
    [1, 1, 1, 1, 1, 0, 1, 1],  # Retrieved 10
])

# Time used
time_data = np.array([
    [9448, 530, 618, 0, 1066, 0, 0, 1927],  # Retrieved 0
    [2267, 560, 0, 2590, 732, 0, 0, 1913],  # Retrieved 1
    [4303, 508, 346, 2639, 712, 0, 1721, 1869],  # Retrieved 3
    [4089, 524, 370, 3699, 701, 2979, 800, 2066],  # Retrieved 5 (baseline)
    [2099, 643, 218, 3077, 699, 0, 939, 1928],  # Retrieved 10
])

# Token used
token_data = np.array([
    [148118, 26141, 89737, 0, 70036, 0, 0, 16065],  # Retrieved 0
    [31443, 28080, 0, 61846, 25730, 0, 0, 17271],  # Retrieved 1
    [70839, 28758, 51204, 69458, 27429, 0, 50466, 23682],  # Retrieved 3
    [79887, 30293, 54606, 72863, 29271, 199127, 14367, 25846],  # Retrieved 5 (baseline)
    [40577, 78403, 29467, 90535, 34600, 0, 23045, 28873],  # Retrieved 10
])

# Calculate success rates
success_rates = np.mean(success_data, axis=1)

# Calculate relative times compared to retrieved 5 (baseline)
baseline_time = time_data[3]  # Retrieved 5
relative_times = []

for i in range(len(retrieval_counts)):
    relative_time_per_task = []
    for task in range(tasks):
        # Only include tasks where both current and baseline succeeded
        if success_data[i][task] > 0 and success_data[3][task] > 0 and time_data[i][task] > 0:
            relative_time_per_task.append(time_data[i][task] / baseline_time[task])
    
    # Calculate average of relative times for tasks that succeeded in both
    if relative_time_per_task:
        relative_times.append(np.mean(relative_time_per_task))
    else:
        relative_times.append(0)

# Calculate relative tokens compared to retrieved 5 (baseline)
baseline_token = token_data[3]  # Retrieved 5
relative_tokens = []

for i in range(len(retrieval_counts)):
    relative_token_per_task = []
    for task in range(tasks):
        # Only include tasks where both current and baseline succeeded
        if success_data[i][task] > 0 and success_data[3][task] > 0 and token_data[i][task] > 0:
            relative_token_per_task.append(token_data[i][task] / baseline_token[task])
    
    # Calculate average of relative tokens for tasks that succeeded in both
    if relative_token_per_task:
        relative_tokens.append(np.mean(relative_token_per_task))
    else:
        relative_tokens.append(0)

# Calculate token efficiency and time efficiency (reciprocals of relative metrics)
# Avoid division by zero
time_efficiency = [1/rt if rt > 0 else 0 for rt in relative_times]
token_efficiency = [1/rt if rt > 0 else 0 for rt in relative_tokens]

# Set up the plot
fig, ax = plt.subplots(figsize=(12, 8), dpi=600)

# Positions for the bars
x = np.arange(len(retrieval_counts))
width = 0.25  # the width of the bars

# Plotting the bars
rects1 = ax.bar(x - width, success_rates, width, label='Success Rate', color='#F1FEC6')
rects2 = ax.bar(x, time_efficiency, width, label='Time Efficiency', color='#F39B6D')
rects3 = ax.bar(x + width, token_efficiency, width, label='Token Efficiency', color='#7BB2D9')

# Add horizontal line at y=1 for baseline reference
ax.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Baseline (5 Docs Retrieved)')

# Increase font size and make bold for all elements
ax.set_xlabel('Number of Retrieved Documents', fontsize=16, fontweight='bold', labelpad=10)
ax.set_ylabel('Performance Metrics', fontsize=16, fontweight='bold', labelpad=10)
ax.set_title('Performance Metrics Across Different Retrieval Counts', fontsize=20, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f'{count}' for count in retrieval_counts], fontsize=14, fontweight='bold')
ax.tick_params(axis='y', labelsize=14)  # Make y-axis tick labels larger

# Make legend text larger and bold
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=4, fontsize=14, frameon=True)

# Add grid
ax.grid(axis='y', linestyle='--', alpha=0.3)

# Add value annotations with larger bold font
def add_labels(rects):
    for rect in rects:
        height = rect.get_height()
        if height > 0:
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=12, fontweight='bold')

add_labels(rects1)
add_labels(rects2)
add_labels(rects3)

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)

# Print the calculated values for reference
print("Success Rates:", success_rates)
print("Time Efficiency:", time_efficiency)
print("Token Efficiency:", token_efficiency)

plt.savefig("./retrieved_number_plots.pdf")