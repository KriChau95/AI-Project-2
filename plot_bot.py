import pandas as pd
import matplotlib.pyplot as plt

# Load data
bot1 = pd.read_csv('bot_2_results.txt', skipinitialspace=True)
# You can switch to bot2 or the moving versions by replacing the file name above.

# Extract alpha and metrics
alpha = bot1['Alpha']
metrics = ['Avg Blocked Cell Detects', 'Avg Space Rat Pings', 'Avg Movements', 'Avg Timesteps']
colors = ['tab:blue', 'tab:orange', 'tab:red', 'tab:green']

# Create the figure and axis
plt.figure(figsize=(12, 7))

# Plot all metrics on the same plot
for i, metric in enumerate(metrics):
    plt.plot(alpha, bot1[metric], label=metric, color=colors[i], marker='o')

plt.title('Bot 2 Metrics vs Alpha')
plt.xlabel('Alpha')
plt.ylabel('Metric Value')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
