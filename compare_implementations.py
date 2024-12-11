import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read both CSV files
df1 = pd.read_csv('results/bfs1_results.csv')  # baseline
df2 = pd.read_csv('results/bfs2_results.csv')  # prefix sum

# Add a column to identify the implementation
df1['implementation'] = 'Baseline'
# df2['implementation'] = 'Prefix Sum'
# df2['implementation'] = 'Shared Memory'
df2['implementation'] = 'Random Branching Graphs'

# Combine the dataframes
combined_df = pd.concat([df1, df2])

# Calculate average times for GPU implementations only
gpu_data = combined_df[combined_df['algorithm'] == 'GPU']
avg_times = gpu_data.groupby(['graph_size', 'implementation'])['time_ms'].mean().reset_index()

# Create a grouped bar plot
plt.figure(figsize=(12, 6))
sns.barplot(data=avg_times, x='graph_size', y='time_ms', hue='implementation')

plt.title('Average GPU Execution Time: Baseline vs Random Branching Graphs')
plt.xlabel('Graph Size')
plt.ylabel('Average Time (ms)')
plt.yscale('log')
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.legend(title='Implementation')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the plot
plt.savefig('results/implementation_comparison.png')
plt.close()

# Print average speedup
print("\nAvailable implementations:", avg_times['implementation'].unique())

baseline_times = avg_times[avg_times['implementation'] == 'Baseline']['time_ms'].values
other_times = avg_times[avg_times['implementation'] == 'Random Branching Graphs']['time_ms'].values
speedup = baseline_times / other_times
print("\nAverage speedup for each graph size:")
for size, speedup in zip(avg_times['graph_size'].unique(), speedup):
    print(f"Graph size {size}: {speedup:.2f}x") 