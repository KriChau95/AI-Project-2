import matplotlib.pyplot as plt

# Initialize lists
alpha = []
blocked = []
pings = []
movements = []
timesteps = []

# Read and parse the file
with open('bot_2_results.txt', 'r') as file:
    lines = file.readlines()

# Skip header and parse values
for line in lines[1:]:
    parts = line.strip().split(',')
    alpha.append(float(parts[0]))
    blocked.append(float(parts[1]))
    pings.append(float(parts[2]))
    movements.append(float(parts[3]))
    timesteps.append(float(parts[4]))

# Compute y-axis range for consistent scale
y_min = 0
y_max = max(max(blocked), max(pings), max(movements), max(timesteps))
y_max_padded = y_max * 1.1

# Plot the metrics
plt.figure(figsize=(10, 6))
plt.plot(alpha, blocked, marker='o', linestyle='-', label='Avg Blocked Cell Detects')
plt.plot(alpha, pings, marker='s', linestyle='-', label='Avg Space Rat Pings')
plt.plot(alpha, movements, marker='^', linestyle='-', label='Avg Movements')
plt.plot(alpha, timesteps, marker='x', linestyle='-', label='Avg Timesteps')
plt.xlabel("Alpha Values")
plt.ylabel("Average Counts")
plt.title("Bot 1 Performance Metrics vs Alpha Values")
plt.ylim(y_min, y_max_padded)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()