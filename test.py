# Importing libraries for randomness, data structures, and data visualization
import os
from ship import *
import matplotlib.pyplot as plt
import random
from collections import defaultdict, deque
import copy

random.seed(42)

# Create the txt files to store results for each bot
bot_1_results_file = "bot_1_results.txt"
bot_2_results_file = "bot_2_results.txt"

# Specify number of ships to create
num_ships = 50
num_iterations = 10

# Initialize array to store num_ships ships
ships = []

# Create num_ships ships by calling init_ship method and add them all to ships array
for i in range(num_ships):
    info = init_ship(30)
    ships.append(info)

# Create dictionaries to store results
# Key = alpha value
# Value = list of result parameters
bot_1_results = defaultdict(list)
bot_2_results = defaultdict(list)

alpha_values = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20]

# # Test Bot 1
# with open(bot_1_results_file, "w") as f:
#     f.write("Alpha, Avg Blocked Cell Detects, Avg Space Rat Pings, Avg Movements, Avg Timesteps\n")
    
#     for alpha in alpha_values:
#         print(f"Testing Bot 1 with alpha = {alpha}")
        
#         avg_num_blocked_cell_detects = 0
#         avg_num_space_rat_pings = 0
#         avg_num_movements = 0
#         avg_timesteps = 0

#         for i in range(len(ships)):
#             for _ in range(num_iterations):
#                 print(f"Testing for ship {i}, iteration {_}, at alpha {alpha}")
#                 visualize = False
#                 curr_ship = copy.deepcopy(ships[i])
#                 # Capture all four returned values
#                 num_blocked_cell_detects, num_space_rat_pings, num_movements, timesteps = bot1(curr_ship, visualize, alpha)
                
#                 avg_num_blocked_cell_detects += num_blocked_cell_detects
#                 avg_num_space_rat_pings += num_space_rat_pings
#                 avg_num_movements += num_movements
#                 avg_timesteps += timesteps
        
#         # Calculate averages
#         total_runs = len(ships) * num_iterations
#         avg_num_blocked_cell_detects /= total_runs
#         avg_num_space_rat_pings /= total_runs
#         avg_num_movements /= total_runs
#         avg_timesteps /= total_runs

#         bot_1_results[alpha] = [avg_num_blocked_cell_detects, avg_num_space_rat_pings, avg_num_movements, avg_timesteps]
        
#         # Write all four metrics to the file
#         f.write(f"{alpha}, {avg_num_blocked_cell_detects}, {avg_num_space_rat_pings}, {avg_num_movements}, {avg_timesteps}\n")


# Test Bot 2
with open(bot_2_results_file, "w") as f:
    f.write("Alpha, Avg Blocked Cell Detects, Avg Space Rat Pings, Avg Movements, Avg Timesteps\n")
    
    for alpha in alpha_values:
        print(f"Testing Bot 2 with alpha = {alpha}")
        
        avg_num_blocked_cell_detects = 0
        avg_num_space_rat_pings = 0
        avg_num_movements = 0
        avg_timesteps = 0

        for i in range(len(ships)):
            for _ in range(num_iterations):
                print(f"Testing for ship {i}, iteration {_}, at alpha {alpha}")
                visualize = False
                curr_ship = copy.deepcopy(ships[i])
                # Capture all four returned values
                num_blocked_cell_detects, num_space_rat_pings, num_movements, timesteps = bot2_3(curr_ship, visualize, alpha)
                
                avg_num_blocked_cell_detects += num_blocked_cell_detects
                avg_num_space_rat_pings += num_space_rat_pings
                avg_num_movements += num_movements
                avg_timesteps += timesteps
        
        # Calculate averages
        total_runs = len(ships) * num_iterations
        avg_num_blocked_cell_detects /= total_runs
        avg_num_space_rat_pings /= total_runs
        avg_num_movements /= total_runs
        avg_timesteps /= total_runs

        bot_2_results[alpha] = [avg_num_blocked_cell_detects, avg_num_space_rat_pings, avg_num_movements, avg_timesteps]
        
        # Write all four metrics to the file
        f.write(f"{alpha}, {avg_num_blocked_cell_detects}, {avg_num_space_rat_pings}, {avg_num_movements}, {avg_timesteps}\n")