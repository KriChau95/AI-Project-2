import pandas as pd
import matplotlib.pyplot as plt

# Load the data
bot1 = pd.read_csv('bot_1_results.txt', skipinitialspace=True)
bot2 = pd.read_csv('bot_2_results.txt', skipinitialspace=True)
bot1_moving = pd.read_csv('bot_1_moving_rat_results.txt', skipinitialspace=True)
bot2_moving = pd.read_csv('bot_2_moving_rat_results.txt', skipinitialspace=True)



# Metrics to plot
metrics = ['Avg Blocked Cell Detects', 'Avg Space Rat Pings', 'Avg Movements', 'Avg Timesteps']
colors = ['tab:blue', 'tab:orange', 'tab:red', 'tab:green', 'tab:purple']

# Plot each metric in its own figure
for metric in metrics:
    plt.figure(figsize=(10, 6))
    plt.plot(bot1['Alpha'], bot1[metric], label='Bot 1', color=colors[0], marker='o')
    plt.plot(bot2['Alpha'], bot2[metric], label='Bot 2 with cost = 225', color=colors[2], marker='x')
    plt.plot(bot1_moving['Alpha'], bot1_moving[metric], label='Bot 1 moving', color=colors[3], marker='x')
    plt.plot(bot2_moving['Alpha'], bot2_moving[metric], label='Bot 2 moving', color=colors[4], marker='x')


    plt.title(f'{metric} vs Alpha')
    plt.xlabel('Alpha')
    plt.ylabel(metric)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
