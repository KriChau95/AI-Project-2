# Importing libraries for randomness, data structures, and data visualization
import os
from ship import *
import matplotlib.pyplot as plt
import random
from collections import defaultdict, deque
import copy

# Create the txt files to store results for each bot
bot_1_results_file = "bot_1_results.txt"

# Set the random seed
random.seed(23) 

# Specify number of ships to create
num_ships = 100

# Initialize array to store num_ships ships
ships = []

# Create num_ships ships by calling int_ship method and add them all to ships array
for i in range(num_ships):
    info = init_ship(30)
    ships.append(info)

# Create dictionaries to store results
# Key = alpha value
# Value = list of result parameters
bot_1_results = defaultdict(list)

alpha_values = [0, 0.02, 0.04, 0.06, 0.1, 0.12, 0.14, 0.16, 0.18, 0.20, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0]
with open(bot_1_results_file, "w") as f:
    f.write("Alpha, Avg Blocked Cell Detects, Avg Space Rat Pings, Avg Movements\n")
    
    for alpha in alpha_values:

        print(f"Testing alpha = {alpha}")

        avg_num_blocked_cell_detects = 0
        avg_num_space_rat_pings = 0
        avg_num_movements = 0

        for i in range(len(ships)):
            print(alpha, i)
            visualize = False
            curr_ship = copy.deepcopy(ships[i])
            num_blocked_cell_detects, num_space_rat_pings, num_movements = bot1(curr_ship, visualize, alpha)
            
            avg_num_blocked_cell_detects += num_blocked_cell_detects
            avg_num_space_rat_pings += num_space_rat_pings
            avg_num_movements += num_movements
        
        avg_num_blocked_cell_detects /= len(ships)
        avg_num_space_rat_pings /= len(ships)
        avg_num_movements /= len(ships)

        bot_1_results[alpha] = [avg_num_blocked_cell_detects, avg_num_space_rat_pings, avg_num_movements]
        
        f.write(f"{alpha}, {avg_num_blocked_cell_detects}, {avg_num_space_rat_pings}, {avg_num_movements}\n")