# Importing libraries for randomness, data structures, and data visualization
import os
import ship
import ship_moving
import matplotlib.pyplot as plt
import random
from collections import defaultdict, deque
import copy

random.seed(42)

# Create the txt files to store results for each bot
bot_1_results_file = "bot_1_results.txt"
bot_2_results_file = "bot_2_results.txt"

bot_1_moving_rat_results_file = "bot_1_moving_rat_results.txt"
bot_2_moving_rat_results_file = "bot_2_moving_rat_results.txt"

# Specify number of ships to create
num_ships = 10
num_iterations = 10

# Initialize array to store num_ships ships
ships = []

# Create num_ships ships by calling init_ship method and add them all to ships array
for i in range(num_ships):
    info = ship.init_ship(30)
    ships.append(info)

# Create dictionaries to store results
# Key = alpha value
# Value = list of result parameters
bot_1_results = defaultdict(list)
bot_2_results = defaultdict(list)
bot_1_moving_rat_results = defaultdict(list)
bot_2_moving_rat_results = defaultdict(list)

# list of alpha values to test
alpha_values = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30]

# Test Bot 1 Stationary Rat Case
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
#                 num_blocked_cell_detects, num_space_rat_pings, num_movements, timesteps = ship.bot1(curr_ship, visualize, alpha)
                
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


# # Test Bot 2 Stationary Rat Case
# with open(bot_2_results_file, "w") as f:
#     f.write("Alpha, Avg Blocked Cell Detects, Avg Space Rat Pings, Avg Movements, Avg Timesteps\n")
    
#     for alpha in alpha_values:
#         print(f"Testing Bot 2 Stationary with alpha = {alpha}")
        
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
#                 num_blocked_cell_detects, num_space_rat_pings, num_movements, timesteps = ship.bot2(curr_ship, visualize, alpha)
                
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

#         bot_2_results[alpha] = [avg_num_blocked_cell_detects, avg_num_space_rat_pings, avg_num_movements, avg_timesteps]
        
#         # Write all four metrics to the file
#         f.write(f"{alpha}, {avg_num_blocked_cell_detects}, {avg_num_space_rat_pings}, {avg_num_movements}, {avg_timesteps}\n")


# # # Test Bot 1 Moving Rat Case
# with open(bot_1_moving_rat_results_file, "w") as f:
    
#     f.write("Alpha, Avg Blocked Cell Detects, Avg Space Rat Pings, Avg Movements, Avg Timesteps\n")
    
#     for alpha in alpha_values:
#         print(f"Testing Bot 1 Moving with alpha = {alpha}")
        
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
#                 num_blocked_cell_detects, num_space_rat_pings, num_movements, timesteps = ship_moving.bot1_2(curr_ship, visualize, alpha)
                
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

#         bot_1_moving_rat_results[alpha] = [avg_num_blocked_cell_detects, avg_num_space_rat_pings, avg_num_movements, avg_timesteps]
        
#         # Write all four metrics to the file
#         f.write(f"{alpha}, {avg_num_blocked_cell_detects}, {avg_num_space_rat_pings}, {avg_num_movements}, {avg_timesteps}\n")


# Test Bot 2 Moving Rat Case
with open(bot_2_moving_rat_results_file, "w") as f:
    f.write("Alpha, Avg Blocked Cell Detects, Avg Space Rat Pings, Avg Movements, Avg Timesteps\n")
    
    for alpha in alpha_values:
        print(f"Testing Bot 2 Moving with alpha = {alpha}")
        
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
                num_blocked_cell_detects, num_space_rat_pings, num_movements, timesteps = ship_moving.bot2_2(curr_ship, visualize, alpha)
                
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