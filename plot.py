import matplotlib.pyplot as plt
import csv

# File containing the data
bot_1_results_file = "bot_1_results.txt"

# Lists to store parsed data
alpha_values = []
avg_blocked_cell_detects = []
avg_space_rat_pings = []
avg_movements = []

# Read data from the file
with open(bot_1_results_file, "r") as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header line
    for row in reader:
        alpha_values.append(float(row[0]))
        avg_blocked_cell_detects.append(float(row[1]))
        avg_space_rat_pings.append(float(row[2]))
        avg_movements.append(float(row[3]))

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(alpha_values, avg_blocked_cell_detects, marker='o', linestyle='-', label='Avg Blocked Cell Detects')
plt.plot(alpha_values, avg_space_rat_pings, marker='s', linestyle='-', label='Avg Space Rat Pings')
plt.plot(alpha_values, avg_movements, marker='^', linestyle='-', label='Avg Movements')

# Formatting the plot
plt.xlabel("Alpha Values")
plt.ylabel("Average Counts")
plt.title("Performance Metrics vs Alpha Values")
plt.legend()
plt.grid(True)
plt.show()
