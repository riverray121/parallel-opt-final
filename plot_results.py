import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV file
df = pd.read_csv('results/bfs_results.csv')

# Create plots for each graph size
plt.figure(figsize=(15, 10))

# Plot time comparison for each graph size
for size in df['graph_size'].unique():
    plt.figure(figsize=(10, 6))
    data = df[df['graph_size'] == size]
    sns.lineplot(data=data, x='branching_factor', y='time_ms', hue='algorithm', marker='o')

    plt.title(f'BFS Performance Comparison - Graph Size {size}')
    plt.xlabel('Branching Factor')
    plt.ylabel('Time (ms)')
    plt.yscale('log')
    plt.grid(True)
    plt.savefig(f'results/performance_size_{size}.png')
    plt.close()

# Create summary plot
plt.figure(figsize=(12, 8))
sns.lineplot(data=df, x='branching_factor', y='time_ms', hue='algorithm', style='graph_size', markers=True)
plt.title('BFS Performance Comparison - All Sizes')
plt.xlabel('Branching Factor')
plt.ylabel('Time (ms)')
plt.yscale('log')
plt.grid(True)
plt.savefig('results/performance_summary.png')
plt.close() 