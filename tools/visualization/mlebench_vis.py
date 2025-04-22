import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
import matplotlib.font_manager as fm

# Set global font to be bold
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 14  # Increase base font size

# Function to create a radar chart
def radar_chart(fig, data, categories, groups, colors, title):
    # Number of variables
    N = len(categories)
    
    # What will be the angle of each axis in the plot
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Initialize the plot
    ax = fig.add_subplot(111, polar=True)
    
    # Improve label positioning
    label_angles = angles[:-1]  # Drop the last angle which is a repeat
    labels = []
    for i, cat in enumerate(categories):
        angle_deg = np.rad2deg(label_angles[i])
        # Adjust text alignment based on position
        if angle_deg == 0:
            ha, va = "center", "center"
        elif 0 < angle_deg < 180:
            ha, va = "left", "bottom"
        elif angle_deg == 180:
            ha, va = "center", "top"
        else:
            ha, va = "right", "bottom"
        
        # Set label closer to the graph (changed from 1.2 to 1.1)
        ax.text(label_angles[i], 1.1, cat, 
                size=20, color='#333333',  # Increased font size
                weight='bold',  # Make text bold
                horizontalalignment=ha, 
                verticalalignment=va)
    
    # Remove default xticks
    plt.xticks([])
    
    # Draw ylabels (removed for cleaner look)
    ax.set_rlabel_position(0)
    plt.yticks([], [])
    plt.ylim(0, 1)  # This will be overridden by individual scales
    
    # Compute max values for each category for normalization
    max_values = data.max(axis=1)
    
    # Plot data
    for i, group in enumerate(groups):
        # Get the values and normalize them
        raw_values = data[group].values.flatten().tolist()
        # Normalize each value by its category's maximum (with a small buffer)
        # FIX: Use iloc instead of direct indexing to avoid FutureWarning
        values = [val / (max_values.iloc[j] * 1.1) for j, val in enumerate(raw_values)]
        values += values[:1]  # Close the loop
        
        ax.plot(angles, values, linewidth=3, linestyle='solid', color=colors[i], label=group)  # Increased linewidth
        ax.fill(angles, values, color=colors[i], alpha=0.25)
    
    # We'll handle the legend in the main code instead
    # Not adding a legend here to avoid duplicates
    
    plt.title(title, size=24, color="#333333", y=1.02, weight='bold')  # Larger, bold title
    
    # Remove the circular grid and spines
    ax.grid(color='#AAAAAA', linestyle='--', alpha=0.7, linewidth=1.5)  # Thicker grid lines
    ax.spines['polar'].set_visible(False)
    
    return ax

# Create some random data (better looking random numbers for visualization)
np.random.seed(42)  # For reproducibility

# Create a DataFrame
categories = ['Rank', '#Gold', '#Silver+', '#Bronze+', '#Median+', 'Success']
groups = ['Auto^2ML', 'AIDE', 'MLAB', 'OpenDevin']

# Create somewhat balanced data that looks meaningful
data = pd.DataFrame({
    'Auto^2ML': [2.5, 4, 5, 6, 12, 0.8],
    'AIDE': [2.5, 3, 4, 5, 8, 0.82],
    'MLAB': [2.5, 0, 1, 1, 1, 0.68],
    'OpenDevin': [2.5, 1, 1, 2, 5, 0.68]
}, index=categories)

# Set up the figure with a light background and proper size
fig = plt.figure(figsize=(14, 12), facecolor="#f9f9f9")  # Increased figure size even more

# Define a pleasing color palette
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# Create the radar chart
ax = radar_chart(fig, data, categories, groups, colors, "MLEBench Results")

# Add the actual values as text annotations for each axis with larger, bold font
for i, category in enumerate(categories):
    angle = i / float(len(categories)) * 2 * np.pi
    # Display the range for each category (moved closer to graph)
    max_val = data.loc[category].max() * 1.05  # 5% buffer
    ax.text(angle, 0.6, f'[0-{max_val:.1f}]',  # Changed from 0.5 to 0.6 to move closer to edge
            horizontalalignment='center',
            verticalalignment='center',
            size=20, color='#555555',
            weight='bold')

# Create a truly larger legend with custom properties
legend = plt.legend(loc='lower right', bbox_to_anchor=(1.1, 0.0), frameon=True, 
                    fontsize=24, prop={'weight': 'bold'})

# Make the legend visually larger and more prominent
legend.get_frame().set_linewidth(2)  # Thicker border
legend.get_frame().set_edgecolor('#333333')  # Darker border
plt.setp(legend.get_texts(), fontsize=24)  # Ensure text is large

# FIX: Use legend.get_lines() instead of legendHandles
for handle in legend.get_lines():
    handle.set_linewidth(6.0)  # Make lines thicker
    
# Adjust spacing between legend items for better visibility
plt.rcParams['legend.labelspacing'] = 1.2  # Vertical space between labels

# Adjust layout and save
plt.tight_layout()
plt.savefig('mlebench_vis.pdf', dpi=300, bbox_inches='tight', facecolor="#f9f9f9")