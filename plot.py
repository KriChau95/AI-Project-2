import matplotlib.pyplot as plt
import csv

# Files containing the data
bot_1_results_file = "bot_1_results.txt"
bot_2_results_file = "bot_2_results.txt"

# Function to read data from a file
def read_results(file_path):
    alpha_values = []
    avg_blocked_cell_detects = []
    avg_space_rat_pings = []
    avg_movements = []
    avg_timesteps = []
    
    with open(file_path, "r") as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header line
        for row in reader:
            alpha_values.append(float(row[0]))
            avg_blocked_cell_detects.append(float(row[1]))
            avg_space_rat_pings.append(float(row[2]))
            avg_movements.append(float(row[3]))
            avg_timesteps.append(float(row[4]))  # New column for timesteps
    
    return alpha_values, avg_blocked_cell_detects, avg_space_rat_pings, avg_movements, avg_timesteps

# Read data for bot1
alpha_1, blocked_1, pings_1, movements_1, timesteps_1 = read_results(bot_1_results_file)

# Read data for bot2
alpha_2, blocked_2, pings_2, movements_2, timesteps_2 = read_results(bot_2_results_file)

# Determine the common y-axis limits
all_blocked = blocked_1 + blocked_2
all_pings = pings_1 + pings_2
all_movements = movements_1 + movements_2
all_timesteps = timesteps_1 + timesteps_2
y_min = 0  # Force y-axis to start at 0
y_max = max(max(all_blocked), max(all_pings), max(all_movements), max(all_timesteps))

# Add padding to y_max (e.g., 10% of the max value) for better visibility
padding = y_max * 0.1
y_max_padded = y_max + padding

# Create a figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)  # sharey=True ensures same y-scale

# Plot for Bot 1
ax1.plot(alpha_1, blocked_1, marker='o', linestyle='-', label='Avg Blocked Cell Detects')
ax1.plot(alpha_1, pings_1, marker='s', linestyle='-', label='Avg Space Rat Pings')
ax1.plot(alpha_1, movements_1, marker='^', linestyle='-', label='Avg Movements')
ax1.plot(alpha_1, timesteps_1, marker='x', linestyle='-', label='Avg Timesteps')  # New plot for timesteps
ax1.set_xlabel("Alpha Values")
ax1.set_ylabel("Average Counts")
ax1.set_title("Bot 1 Performance Metrics vs Alpha Values")
ax1.legend()
ax1.grid(True)
ax1.set_ylim(y_min, y_max_padded)  # Set y-axis from 0 to padded max

# Plot for Bot 2
ax2.plot(alpha_2, blocked_2, marker='o', linestyle='-', label='Avg Blocked Cell Detects')
ax2.plot(alpha_2, pings_2, marker='s', linestyle='-', label='Avg Space Rat Pings')
ax2.plot(alpha_2, movements_2, marker='^', linestyle='-', label='Avg Movements')
ax2.plot(alpha_2, timesteps_2, marker='x', linestyle='-', label='Avg Timesteps')  # New plot for timesteps
ax2.set_xlabel("Alpha Values")
# ax2.set_ylabel("Average Counts")  # Not needed due to sharey=True
ax2.set_title("Bot 2 Performance Metrics vs Alpha Values")
ax2.legend()
ax2.grid(True)
ax2.set_ylim(y_min, y_max_padded)  # Set y-axis from 0 to padded max

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()