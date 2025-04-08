# ship.py represents all the information to simulate the space vessel and contains bot functions for each bot

# Importing libraries for randomness, data structures, and data visualization
import random
import numpy as np
import heapq
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors 
import matplotlib.animation as animation
from collections import deque, defaultdict
import copy
import math
from visualize import *

# setting up global variables that are used for adjacency in searches
global directions
directions = [(0,1), (0,-1), (1,0), (-1,0)] # array to store adjacent directions needed during various traversal
global diagonal_directions
diagonal_directions = [(0,1), (0,-1), (1,0), (-1,0), (-1,1), (-1,-1), (1,1), (1,-1)]

# the ping function takes in info - the current state of the ship, and alpha which dictates ping probability
# based on the current bot position and the current rat position, it simulates and returns a boolean representing whether or not a ping occurred
# 3 possible outputs:
# 1. "Found" if bot position == rat position
# 2. "True" if bot posiiton != rat position and hear a ping
# 3. "False" if bot position != rat position and not hear a ping
# Parameters: 
#   - info - state of ship to extract bot and rat position
#   - alpha - parameter for probability of ping
def ping(info, alpha):

    bot_r, bot_c = info['bot']
    rat_r, rat_c = info['rat']

    if bot_r == rat_r and bot_c == rat_c:
        return "Found"

    def heuristic():
        return abs(bot_r - rat_r) + abs(bot_c - rat_c)
    
    prob_ping = math.e ** (-alpha * (heuristic() - 1))
    
    if random.uniform(0,1) < prob_ping:
        return "True"
    else:
        return "False"

# helper function when determining initial bot position that creates two useful structures to store information abotu blocked neighbors
# neighbor_map: 2D array of same size as ship
# neighbor_map[r][c] = 
#   -1 : if closed cell
#   n = number of blocked neighbors : if open cell
# blocked_neighbors: dictionary such that key = number of blocked neighbors, value = list of all tuples representing coordinates with that many blocked neighbors
def create_neighbor_map(ship):
    blocked_cells = defaultdict(list)
    d = len(ship)
    neighbor_map = [[-1] * d for _ in range(d)]
    for r in range(1,d-1):
        for c in range(1,d-1):
            if ship[r][c] == 1:
                neighbor_map[r][c] = -1
                continue
            blocked_neighbors = 0
            for dr, dc in diagonal_directions:
                nr,nc = r + dr, c + dc
                if ship[nr][nc] == 1:
                    blocked_neighbors += 1
            neighbor_map[r][c] = blocked_neighbors
            blocked_cells[blocked_neighbors].append((r,c))
    return neighbor_map, blocked_cells

def bot1(info, visualize, alpha):

    # Get initial bot position and ship configuration
    bot_r, bot_c = info['bot']
    ship = info['empty_ship']
    d = len(info['ship']) # dimension of ship grid

    # Initialize current bot position
    curr_r, curr_c = bot_r, bot_c

    # Generate helper variables:
    # - neighbor_map: maps each cell to # of blocked neighbors (2D array)
    # - blocked_neighbors: map from # blocked neighbors to corresponding cell coordinates
    neighbor_map, blocked_neighbors = create_neighbor_map(ship)
    
    # Initialize a set of possible cells based on current blocked neighbor count
    set_possible_cells = {
        (r, c)
        for r in range(len(ship))
        for c in range(len(ship[0]))
        if ship[r][c] == 0
    }

    # Initialize result variables that measure efficiency of rat-finding process
    num_movements = 0
    num_blocked_cell_detects = 0
    num_space_rat_pings = 0
    timesteps = 0  # Tracks total actions (move, blocked cell detects, pings)

    # ----------------- PHASE 1: Localization -----------------
    # Repeat until only one possible bot location remains
    while len(set_possible_cells) > 1:

        # visualize_possible_cells(ship, possible_cells, title = f"possible cells, curr_pos: {curr_r}, {curr_c}")
        
        # Sense blocked neighbors at current position
        num_curr_blocked_ns = neighbor_map[curr_r][curr_c]
        
        # filter possible cells to only include those that match current number of blocked neighbors
        possible_cells = set()
        for cellr, cellc in set_possible_cells:
            if neighbor_map[cellr][cellc] == num_curr_blocked_ns:
                possible_cells.add((cellr, cellc))
        
        # increment blocked_cell_detects and timestamps
        num_blocked_cell_detects += 1
        timesteps += 1

        # key = direction, value = set of cells that are closed if we move from any of the current possible cells to that direction
        direction_c = {(0,1): set(), (0,-1): set(), (-1,0): set(), (1,0): set()}

        # loop through possible_cells set to populate direction_c
        for pcr, pcc in possible_cells:
            for dr, dc in directions:
                nr = pcr + dr
                nc = pcc + dc
                if ship[nr][nc] != 0:
                    direction_c[(dr,dc)].add((pcr,pcc))
        
        # randomly choose a direction and move in that direction
        best_dir = random.choice(directions)
        nr, nc = curr_r + best_dir[0], curr_c + best_dir[1]

        # if the move is successful
        if ship[nr][nc] == 0:  

            # remove all possible cells that had that move being impossible (blocked cell in that direction)
            set_possible_cells = set(possible_cells).difference(direction_c[best_dir])
            
            # move in that direction
            curr_r, curr_c = nr, nc
            
            # modify the set of possible cells so we are considering all possible cells after the movement we just made
            new_set_possible_cells = set()
            for elem_r, elem_c in set_possible_cells:
                new_cell = (elem_r + best_dir[0], elem_c + best_dir[1])
                new_set_possible_cells.add(new_cell)
            set_possible_cells = copy.deepcopy(new_set_possible_cells)
            
            # Update parameters in info to reflect changes 
            info['bot'] = (curr_r, curr_c)  # Update bot position in info
            info['ship'][bot_r][bot_c] = 0  # Clear old position
            info['ship'][curr_r][curr_c] = 2  # Set new position

            # Update bot position in loop
            bot_r, bot_c = curr_r, curr_c 
       
        else:  # Move unsuccessful

            # change set of possible cells to all cells that had impossible movement in that direction
            set_possible_cells = direction_c[best_dir]

        # visualize_possible_cells(ship, set_possible_cells, title = f"possible cells, curr_pos: {curr_r}, {curr_c}, move: {best_dir[0]}, {best_dir[1]}")
        
        num_movements += 1
        timesteps += 1

    # ----------------- PHASE 2: Rat Finding -----------------

    # store current position as only possible cell based on localization
    curr_r, curr_c = set_possible_cells.pop()

    # update info variables to be aligned with new changes and current ship state and bot position
    info['ship'][bot_r][bot_c] = 0
    info['ship'][curr_r][curr_c] = 2
    info['bot'] = (curr_r, curr_c)

    # Sync bot_r, bot_c with new position
    bot_r, bot_c = curr_r, curr_c  

    # Create Probability Map s.t. rat_prob_map[r][c] = probability that rat is at position r,c based on ping information so far
    rat_prob_map = copy.deepcopy(info['empty_ship'])
    
    # Initialize all the probabilities for open cells to be uniformly the same number
    # let's say we have x open cells. At the beginning, P(rat at r,c) = 1/x for all open cells (r,c)
    num_open_cells = 0
    for i in range(len(rat_prob_map)):
        for j in range(len(rat_prob_map)):
            if rat_prob_map[i][j] == 0:  # open
                num_open_cells += 1
            else:
                rat_prob_map[i][j] = -1    
    uniform_prob_i = 1 / num_open_cells

    # store coordinates for all open cells - useful for probability updates in ratfinding process
    open_cells = set()
    for i in range(len(rat_prob_map)):
        for j in range(len(rat_prob_map)):
            if rat_prob_map[i][j] == 0: 
                rat_prob_map[i][j] = uniform_prob_i
                open_cells.add((i,j))

    # empty path which stores the current planned path coordinates of the bot to its next destination on the ship
    path = []

    # Initial ping before the loop
    ping_result = ping(info, alpha)

    # Increment counter variables appropriately
    num_space_rat_pings += 1
    timesteps += 1

    # if we are at the rat, return all the counter variables for all actions
    if ping_result == 'Found':
        return num_blocked_cell_detects, num_space_rat_pings, num_movements, timesteps
    
    # if we are not at the rat, update rat_prob_map
    else:
        
        # rat cannot be at this position because we have not found the rat
        rat_prob_map[bot_r][bot_c] = 0

        # update probability that rat is at any other cell based on whether or not we heard a ping using conditional probability calculations described in write-up
        summed_prob = 0

        if ping_result == 'True':
            for (r,c) in open_cells:
                prob_rat = math.e ** (-alpha * ((abs(r - bot_r) + abs(c - bot_c))-1))
                rat_prob_map[r][c] *= prob_rat
                summed_prob += rat_prob_map[r][c]
            for (r,c) in open_cells:
                rat_prob_map[r][c] /= summed_prob

        elif ping_result == 'False':
            for (r,c) in open_cells:
                rat_prob_map[r][c] *= (1 - math.e ** (-alpha * ((abs(r - bot_r) + abs(c - bot_c))-1)))
                summed_prob += rat_prob_map[r][c]
            for (r,c) in open_cells:
                rat_prob_map[r][c] /= summed_prob

    # found variable that keeps track of whether or not rat has been found
    found = False

    # keep pinging and moving until we find the rat
    while not found:
        
        if visualize: visualize_rat_prob_map(rat_prob_map)
        
        # determine the coordinates of the cell where we think the rat is most likely at based on all the ping results we have done so far
        highest_rat_prob = 0
        highest_rat_prob_cell = (-1,-1)
        for i in range(len(rat_prob_map)):
            for j in range(len(rat_prob_map)):
                if rat_prob_map[i][j] > highest_rat_prob:
                    highest_rat_prob = rat_prob_map[i][j]
                    highest_rat_prob_cell = (i,j)

        # plan a shortest distance path to that cell using A*
        path = astar(info['bot'], info['empty_ship'], highest_rat_prob_cell)

        # commit to that path and move along every cell in the shortest distance path
        for new_r, new_c in path:
            
            # Move
            info['ship'][bot_r][bot_c] = 0  # Clear old position
            info['bot'] = (new_r, new_c) # Update new bot position
            info['ship'][new_r][new_c] = 2  # Set new position in ship
            bot_r, bot_c = new_r, new_c  # Update bot_r, bot_c
            
            # increment counter variables by 1 - move and timesteps
            num_movements += 1
            timesteps += 1
            
            # ping the rat and increment appropriate counter variables
            ping_result = ping(info, alpha)
            num_space_rat_pings += 1
            timesteps += 1

            # if rat is found, we are done; return counter variables
            if ping_result == 'Found':
                return num_blocked_cell_detects, num_space_rat_pings, num_movements, timesteps

            # if rat is not found, update rat_prob_map based on conditional probability approach described in write up
            else:
                summed_prob = 0
                rat_prob_map[bot_r][bot_c] = 0
                if ping_result == 'True':
                    for (r,c) in open_cells:
                        prob_rat = math.e ** (-alpha * ((abs(r - bot_r) + abs(c - bot_c))-1))
                        rat_prob_map[r][c] *= prob_rat
                        summed_prob += rat_prob_map[r][c]
                    for (r,c) in open_cells:
                        rat_prob_map[r][c] /= summed_prob
                elif ping_result == 'False':
                    for (r,c) in open_cells:
                        rat_prob_map[r][c] *= (1 - math.e ** (-alpha * ((abs(r - bot_r) + abs(c - bot_c))-1)))
                        summed_prob += rat_prob_map[r][c]
                    for (r,c) in open_cells:
                        rat_prob_map[r][c] /= summed_prob
            
            if visualize:
                visualize_ship(ship, path, title = "Planned Path")
                visualize_rat_prob_map(rat_prob_map, None, title = f"Rat_prob_map Timestep = {timesteps}")


def bot3(info, visualize, alpha):

    # Get initial bot position and ship configuration
    bot_r, bot_c = info['bot']
    ship = info['empty_ship']
    d = len(info['ship']) # dimension of ship grid

    # Initialize current bot position
    curr_r, curr_c = bot_r, bot_c

    # Generate helper variables:
    # - neighbor_map: maps each cell to # of blocked neighbors (2D array)
    # - blocked_neighbors: map from # blocked neighbors to corresponding cell coordinates
    neighbor_map, blocked_neighbors = create_neighbor_map(ship)
    
    # Initialize a set of possible cells based on current blocked neighbor count
    set_possible_cells = {
        (r, c)
        for r in range(len(ship))
        for c in range(len(ship[0]))
        if ship[r][c] == 0
    }

    # Initialize result variables that measure efficiency of rat-finding process
    num_movements = 0
    num_blocked_cell_detects = 0
    num_space_rat_pings = 0
    timesteps = 0  # Tracks total actions (move, blocked cell detects, pings)

    # ----------------- PHASE 1: Localization -----------------
    # Repeat until only one possible bot location remains
    while len(set_possible_cells) > 1:

        # visualize_possible_cells(ship, possible_cells, title = f"possible cells, curr_pos: {curr_r}, {curr_c}")
        
        # Sense blocked neighbors at current position
        num_curr_blocked_ns = neighbor_map[curr_r][curr_c]
        
        # filter possible cells to only include those that match current number of blocked neighbors
        possible_cells = set()
        for cellr, cellc in set_possible_cells:
            if neighbor_map[cellr][cellc] == num_curr_blocked_ns:
                possible_cells.add((cellr, cellc))
        
        # increment blocked_cell_detects and timestamps
        num_blocked_cell_detects += 1
        timesteps += 1

        # key = direction, value = set of cells that are closed if we move from any of the current possible cells to that direction
        direction_c = {(0,1): set(), (0,-1): set(), (-1,0): set(), (1,0): set()}

        # loop through possible_cells set to populate direction_c
        for pcr, pcc in possible_cells:
            for dr, dc in directions:
                nr = pcr + dr
                nc = pcc + dc
                if ship[nr][nc] != 0:
                    direction_c[(dr,dc)].add((pcr,pcc))
        
        # randomly choose a direction and move in that direction
        best_dir = random.choice(directions)
        nr, nc = curr_r + best_dir[0], curr_c + best_dir[1]

        # if the move is successful
        if ship[nr][nc] == 0:  

            # remove all possible cells that had that move being impossible (blocked cell in that direction)
            set_possible_cells = set(possible_cells).difference(direction_c[best_dir])
            
            # move in that direction
            curr_r, curr_c = nr, nc
            
            # modify the set of possible cells so we are considering all possible cells after the movement we just made
            new_set_possible_cells = set()
            for elem_r, elem_c in set_possible_cells:
                new_cell = (elem_r + best_dir[0], elem_c + best_dir[1])
                new_set_possible_cells.add(new_cell)
            set_possible_cells = copy.deepcopy(new_set_possible_cells)
            
            # Update parameters in info to reflect changes 
            info['bot'] = (curr_r, curr_c)  # Update bot position in info
            info['ship'][bot_r][bot_c] = 0  # Clear old position
            info['ship'][curr_r][curr_c] = 2  # Set new position

            # Update bot position in loop
            bot_r, bot_c = curr_r, curr_c 
       
        else:  # Move unsuccessful

            # change set of possible cells to all cells that had impossible movement in that direction
            set_possible_cells = direction_c[best_dir]

        # visualize_possible_cells(ship, set_possible_cells, title = f"possible cells, curr_pos: {curr_r}, {curr_c}, move: {best_dir[0]}, {best_dir[1]}")
        
        num_movements += 1
        timesteps += 1

    # ----------------- PHASE 2: Rat Finding -----------------

    # store current position as only possible cell based on localization
    curr_r, curr_c = set_possible_cells.pop()

    # update info variables to be aligned with new changes and current ship state and bot position
    info['ship'][bot_r][bot_c] = 0
    info['ship'][curr_r][curr_c] = 2
    info['bot'] = (curr_r, curr_c)

    # Sync bot_r, bot_c with new position
    bot_r, bot_c = curr_r, curr_c  

    # Create Probability Map s.t. rat_prob_map[r][c] = probability that rat is at position r,c based on ping information so far
    rat_prob_map = copy.deepcopy(info['empty_ship'])
    
    # Initialize all the probabilities for open cells to be uniformly the same number
    # let's say we have x open cells. At the beginning, P(rat at r,c) = 1/x for all open cells (r,c)
    num_open_cells = 0
    for i in range(len(rat_prob_map)):
        for j in range(len(rat_prob_map)):
            if rat_prob_map[i][j] == 0:  # open
                num_open_cells += 1
            else:
                rat_prob_map[i][j] = -1    
    uniform_prob_i = 1 / num_open_cells

    # store coordinates for all open cells - useful for probability updates in ratfinding process
    open_cells = set()
    for i in range(len(rat_prob_map)):
        for j in range(len(rat_prob_map)):
            if rat_prob_map[i][j] == 0: 
                rat_prob_map[i][j] = uniform_prob_i
                open_cells.add((i,j))

    # empty path which stores the current planned path coordinates of the bot to its next destination on the ship
    path = []

    initial_pings = 20
    for i in range(initial_pings):

        # Initial ping before the loop
        ping_result = ping(info, alpha)

        # Increment counter variables appropriately
        num_space_rat_pings += 1
        timesteps += 1

        # if we are at the rat, return all the counter variables for all actions
        if ping_result == 'Found':
            return num_blocked_cell_detects, num_space_rat_pings, num_movements, timesteps
        
        # if we are not at the rat, update rat_prob_map
        else:
            
            # rat cannot be at this position because we have not found the rat
            rat_prob_map[bot_r][bot_c] = 0

            # update probability that rat is at any other cell based on whether or not we heard a ping using conditional probability calculations described in write-up
            summed_prob = 0

            if ping_result == 'True':
                for (r,c) in open_cells:
                    prob_rat = math.e ** (-alpha * ((abs(r - bot_r) + abs(c - bot_c))-1))
                    rat_prob_map[r][c] *= prob_rat
                    summed_prob += rat_prob_map[r][c]
                for (r,c) in open_cells:
                    rat_prob_map[r][c] /= summed_prob

            elif ping_result == 'False':
                for (r,c) in open_cells:
                    rat_prob_map[r][c] *= (1 - math.e ** (-alpha * ((abs(r - bot_r) + abs(c - bot_c))-1)))
                    summed_prob += rat_prob_map[r][c]
                for (r,c) in open_cells:
                    rat_prob_map[r][c] /= summed_prob
                    
        if visualize: visualize_rat_prob_map(rat_prob_map)

    # found variable that keeps track of whether or not rat has been found
    found = False

    # keep pinging and moving until we find the rat
    while not found:
        
        if visualize: visualize_rat_prob_map(rat_prob_map)
        
        # determine the coordinates of the cell where we think the rat is most likely at based on all the ping results we have done so far
        highest_rat_prob = 0
        highest_rat_prob_cell = (-1,-1)
        for i in range(len(rat_prob_map)):
            for j in range(len(rat_prob_map)):
                if rat_prob_map[i][j] > highest_rat_prob:
                    highest_rat_prob = rat_prob_map[i][j]
                    highest_rat_prob_cell = (i,j)

        # plan a shortest distance path to that cell using A*
        path = astar(info['bot'], info['empty_ship'], highest_rat_prob_cell)

        # commit to that path and move along every cell in the shortest distance path
        for new_r, new_c in path:
            
            # Move
            info['ship'][bot_r][bot_c] = 0  # Clear old position
            info['bot'] = (new_r, new_c) # Update new bot position
            info['ship'][new_r][new_c] = 2  # Set new position in ship
            bot_r, bot_c = new_r, new_c  # Update bot_r, bot_c
            
            # increment counter variables by 1 - move and timesteps
            num_movements += 1
            timesteps += 1
            
            # ping the rat and increment appropriate counter variables
            ping_result = ping(info, alpha)
            num_space_rat_pings += 1
            timesteps += 1

            # if rat is found, we are done; return counter variables
            if ping_result == 'Found':
                return num_blocked_cell_detects, num_space_rat_pings, num_movements, timesteps

            # if rat is not found, update rat_prob_map based on conditional probability approach described in write up
            else:
                summed_prob = 0
                rat_prob_map[bot_r][bot_c] = 0
                if ping_result == 'True':
                    for (r,c) in open_cells:
                        prob_rat = math.e ** (-alpha * ((abs(r - bot_r) + abs(c - bot_c))-1))
                        rat_prob_map[r][c] *= prob_rat
                        summed_prob += rat_prob_map[r][c]
                    for (r,c) in open_cells:
                        rat_prob_map[r][c] /= summed_prob
                elif ping_result == 'False':
                    for (r,c) in open_cells:
                        rat_prob_map[r][c] *= (1 - math.e ** (-alpha * ((abs(r - bot_r) + abs(c - bot_c))-1)))
                        summed_prob += rat_prob_map[r][c]
                    for (r,c) in open_cells:
                        rat_prob_map[r][c] /= summed_prob
            
            if visualize:
                visualize_ship(ship, path, title = "Planned Path")
                visualize_rat_prob_map(rat_prob_map, None, title = f"Rat_prob_map Timestep = {timesteps}")

def bot2(info, visualize, alpha):
    # Get initial information about bot and ship
    bot_r, bot_c = info['bot']
    ship = info['empty_ship']
    d = len(info['ship'])

    # Set up variables to help determine bot position
    curr_r, curr_c = bot_r, bot_c
    neighbor_map, blocked_neighbors = create_neighbor_map(ship)
    num_curr_blocked_ns = neighbor_map[curr_r][curr_c]
    possible_cells = set_possible_cells = set(blocked_neighbors[num_curr_blocked_ns])
    prev_dirc = (1,1)
    
    # Initialize result variables that provide insight to rat-finding process
    num_movements = 0
    num_blocked_cell_detects = 0
    num_space_rat_pings = 0
    timesteps = 0  # Tracks total actions (sense, ping, move)

    print("phase 1")

    ## Phase 1: Localization
    while len(possible_cells) > 1:

        #visualize_possible_cells(ship, possible_cells, title = f"possible cells, curr_pos: {curr_r}, {curr_c}")
        # Sense blocked neighbors (alternating action 1)
        num_curr_blocked_ns = neighbor_map[curr_r][curr_c]
        #visualize_neighbor_map(neighbor_map, title = f"NEIGHBOR MAPPPP, curr_pos: {curr_r}, {curr_c}")
        #print("num_curr_blocked_ns", num_curr_blocked_ns)
        possible_cells = set()
        for cellr, cellc in set_possible_cells:
            if neighbor_map[cellr][cellc] == num_curr_blocked_ns:
                possible_cells.add((cellr, cellc))
        num_blocked_cell_detects += 1
        timesteps += 1


        # Determine the most commonly open direction and attempt to move (alternating action 2)
        direction_o = {(0,1): 0, (0,-1): 0, (-1,0): 0, (1,0): 0}
        direction_c = {(0,1): set(), (0,-1): set(), (-1,0): set(), (1,0): set()}

        for pcr, pcc in possible_cells:
            for dr, dc in directions:
                nr = pcr + dr
                nc = pcc + dc
                if ship[nr][nc] == 0:
                    direction_o[(dr,dc)] += 1
                else:
                    direction_c[(dr,dc)].add((pcr,pcc))
        
        best_dir_arr = sorted(direction_o, key=lambda x: direction_o[x])
        best_dir = random.choice(best_dir_arr)
    
        nr, nc = curr_r + best_dir[0], curr_c + best_dir[1]

        if ship[nr][nc] == 0:  # Open, move successful
            set_possible_cells = set(possible_cells).difference(direction_c[best_dir])
            curr_r, curr_c = nr, nc
            num_curr_blocked_ns = neighbor_map[curr_r][curr_c]
            new_set_possible_cells = set()
            for elem_r, elem_c in set_possible_cells:
                new_cell = (elem_r + best_dir[0], elem_c + best_dir[1])
                new_set_possible_cells.add(new_cell)
            set_possible_cells = copy.deepcopy(new_set_possible_cells)
            info['bot'] = (curr_r, curr_c)  # Update bot position in info
            info['ship'][bot_r][bot_c] = 0  # Clear old position
            info['ship'][curr_r][curr_c] = 2  # Set new position
            bot_r, bot_c = curr_r, curr_c  # Update bot_r, bot_c
        else:  # Blocked, no movement
            set_possible_cells = direction_c[best_dir]
        
        #print("best_dir", best_dir)
        num_movements += 1
        
        timesteps += 1

   
    



    #
    ##
    ###
    #### Phase 2: Rat finding
    ###
    ##
    #
    print("phase 2")
    curr_r, curr_c = set_possible_cells.pop()
    info['ship'][bot_r][bot_c] = 0
    info['ship'][curr_r][curr_c] = 2
    info['bot'] = (curr_r, curr_c)
    bot_r, bot_c = curr_r, curr_c  # Sync bot_r, bot_c with new position

    # creating rat prob map
    rat_prob_map = copy.deepcopy(info['empty_ship'])
    num_open_cells = 0
    for i in range(len(rat_prob_map)):
        for j in range(len(rat_prob_map)):
            if rat_prob_map[i][j] == 0:  # open
                num_open_cells += 1
            else:
                rat_prob_map[i][j] = -1    
    uniform_prob_i = 1 / num_open_cells

    open_cells = set()
    for i in range(len(rat_prob_map)):
        for j in range(len(rat_prob_map)):
            if rat_prob_map[i][j] == 0: 
                rat_prob_map[i][j] = uniform_prob_i
                open_cells.add((i,j))

    path = []
    num_pings = 1
    # Initial ping before the loop
    for i in range(num_pings):
        ping_result = ping(info, alpha)
        num_space_rat_pings += 1
        timesteps += 1
        # visualize_rat_prob_map(rat_prob_map,None)
        if ping_result == 'Found':
            return num_blocked_cell_detects, num_space_rat_pings, num_movements, timesteps
        else:
            rat_prob_map[bot_r][bot_c] = 0
            summed_prob = 0
            if ping_result == 'True':
                for (r,c) in open_cells:
                    prob_rat = math.e ** (-alpha * ((abs(r - bot_r) + abs(c - bot_c))-1))
                    rat_prob_map[r][c] *= prob_rat
                    summed_prob += rat_prob_map[r][c]
                for (r,c) in open_cells:
                    rat_prob_map[r][c] /= summed_prob
            elif ping_result == 'False':
                for (r,c) in open_cells:
                    rat_prob_map[r][c] *= (1 - math.e ** (-alpha * ((abs(r - bot_r) + abs(c - bot_c))-1)))
                    summed_prob += rat_prob_map[r][c]
                for (r,c) in open_cells:
                    rat_prob_map[r][c] /= summed_prob
        if visualize: visualize_rat_prob_map(rat_prob_map,None, f"ping = {ping_result}")



    found = False



    while not found:


        if visualize: visualize_rat_prob_map(rat_prob_map)
        highest_rat_prob = 0
        highest_rat_prob_cell = (-1,-1)
        for i in range(len(rat_prob_map)):
            for j in range(len(rat_prob_map)):
                "come back here"
                # manhattan_dist = abs(i - bot_r) + abs(j - bot_c)
                temp = rat_prob_map[i][j] 
                if temp > highest_rat_prob:
                    highest_rat_prob = temp
                    highest_rat_prob_cell = (i,j)
        
                    
        path = astar_with_heuristic(info['bot'], rat_prob_map, info['empty_ship'], highest_rat_prob_cell)
        steps_towards_before_recalc = 15
        step = 0
        for new_r, new_c in path[:steps_towards_before_recalc]:
            # Move
            info['ship'][bot_r][bot_c] = 0  # Clear old position
            info['bot'] = (new_r, new_c)
            info['ship'][new_r][new_c] = 2  # Set new position
            bot_r, bot_c = new_r, new_c  # Update bot_r, bot_c
            num_movements += 1
            timesteps += 1
            ping_result = ping(info, alpha)
            num_space_rat_pings += 1

            # if num_movements % steps_before_ping == 0:
            #     for i in range(num_ping):
            timesteps += 1

            if ping_result == 'Found':
                return num_blocked_cell_detects, num_space_rat_pings, num_movements, timesteps
            else:
                summed_prob = 0
                rat_prob_map[bot_r][bot_c] = 0
                if ping_result == 'True':
                    for (r,c) in open_cells:
                        prob_rat = math.e ** (-alpha * ((abs(r - bot_r) + abs(c - bot_c))-1))
                        rat_prob_map[r][c] *= prob_rat
                        summed_prob += rat_prob_map[r][c]
                    for (r,c) in open_cells:
                        rat_prob_map[r][c] /= summed_prob
                elif ping_result == 'False':
                    for (r,c) in open_cells:
                        rat_prob_map[r][c] *= (1 - math.e ** (-alpha * ((abs(r - bot_r) + abs(c - bot_c))-1)))
                        summed_prob += rat_prob_map[r][c]
                    for (r,c) in open_cells:
                        rat_prob_map[r][c] /= summed_prob
            if visualize: visualize_rat_prob_map(rat_prob_map,None, f"ping = {ping_result}")

            step += 1
                
            if visualize:
                visualize_ship(ship, path, title = "Planned Path")
                visualize_rat_prob_map(rat_prob_map, None, title = f"Rat_prob_map Timestep = {timesteps}")



    return num_blocked_cell_detects, num_space_rat_pings, num_movements, timesteps



def astar_with_heuristic(start, rat_prob_map, map, button):
    def heuristic(cell1):
        return abs(cell1[0] - button[0]) + abs(cell1[1] - button[1])  

    d = len(map)
    fringe = []
    heapq.heappush(fringe, (heuristic(start), start))
    total_costs = dict()
    total_costs[start] = 0
    prev = dict()
    prev[start] = None



    while fringe:
        curr = heapq.heappop(fringe)
        if curr[1] == button:
            curr_p = curr[1]
            path = deque()
            while curr_p != None:
                path.appendleft(curr_p)
                curr_p = prev[curr_p]
            return list(path)
        
        r, c = curr[1]
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            child = (nr, nc)
            if 0 <= nr < d and 0 <= nc < d and (map[nr][nc] != 1 and map[nr][nc] != -1):
                cost = total_costs[curr[1]] + 1

                # print(cost, 150 * rat_prob_map[nr][nc])
                cost -= (200 * rat_prob_map[nr][nc])
                cost = max(cost, 0)

                est_total_cost = cost + heuristic(child)
                if child not in total_costs:
                    prev[child] = curr[1]
                    total_costs[child] = cost
                    heapq.heappush(fringe, (est_total_cost, child))
    return []        


def bot2_entropy_experiments(info, visualize, alpha):

    # Get initial bot position and ship configuration
    bot_r, bot_c = info['bot']
    ship = info['empty_ship']
    d = len(info['ship']) # dimension of ship grid

    # Initialize current bot position
    curr_r, curr_c = bot_r, bot_c

    # Generate helper variables:
    # - neighbor_map: maps each cell to # of blocked neighbors (2D array)
    # - blocked_neighbors: map from # blocked neighbors to corresponding cell coordinates
    neighbor_map, blocked_neighbors = create_neighbor_map(ship)
    
    # Initialize a set of possible cells based on current blocked neighbor count
    set_possible_cells = {
        (r, c)
        for r in range(len(ship))
        for c in range(len(ship[0]))
        if ship[r][c] == 0
    }

    # Initialize result variables that measure efficiency of rat-finding process
    num_movements = 0
    num_blocked_cell_detects = 0
    num_space_rat_pings = 0
    timesteps = 0  # Tracks total actions (move, blocked cell detects, pings)

    # ----------------- PHASE 1: Localization -----------------
    # Repeat until only one possible bot location remains
    while len(set_possible_cells) > 1:

        # visualize_possible_cells(ship, possible_cells, title = f"possible cells, curr_pos: {curr_r}, {curr_c}")
        
        # Sense blocked neighbors at current position
        num_curr_blocked_ns = neighbor_map[curr_r][curr_c]
        
        # filter possible cells to only include those that match current number of blocked neighbors
        possible_cells = set()
        for cellr, cellc in set_possible_cells:
            if neighbor_map[cellr][cellc] == num_curr_blocked_ns:
                possible_cells.add((cellr, cellc))
        
        # increment blocked_cell_detects and timestamps
        num_blocked_cell_detects += 1
        timesteps += 1

        # key = direction, value = set of cells that are closed if we move from any of the current possible cells to that direction
        direction_c = {(0,1): set(), (0,-1): set(), (-1,0): set(), (1,0): set()}

        # loop through possible_cells set to populate direction_c
        for pcr, pcc in possible_cells:
            for dr, dc in directions:
                nr = pcr + dr
                nc = pcc + dc
                if ship[nr][nc] != 0:
                    direction_c[(dr,dc)].add((pcr,pcc))
        
        # randomly choose a direction and move in that direction
        best_dir = random.choice(directions)
        nr, nc = curr_r + best_dir[0], curr_c + best_dir[1]

        # if the move is successful
        if ship[nr][nc] == 0:  

            # remove all possible cells that had that move being impossible (blocked cell in that direction)
            set_possible_cells = set(possible_cells).difference(direction_c[best_dir])
            
            # move in that direction
            curr_r, curr_c = nr, nc
            
            # modify the set of possible cells so we are considering all possible cells after the movement we just made
            new_set_possible_cells = set()
            for elem_r, elem_c in set_possible_cells:
                new_cell = (elem_r + best_dir[0], elem_c + best_dir[1])
                new_set_possible_cells.add(new_cell)
            set_possible_cells = copy.deepcopy(new_set_possible_cells)
            
            # Update parameters in info to reflect changes 
            info['bot'] = (curr_r, curr_c)  # Update bot position in info
            info['ship'][bot_r][bot_c] = 0  # Clear old position
            info['ship'][curr_r][curr_c] = 2  # Set new position

            # Update bot position in loop
            bot_r, bot_c = curr_r, curr_c 
       
        else:  # Move unsuccessful

            # change set of possible cells to all cells that had impossible movement in that direction
            set_possible_cells = direction_c[best_dir]

        # visualize_possible_cells(ship, set_possible_cells, title = f"possible cells, curr_pos: {curr_r}, {curr_c}, move: {best_dir[0]}, {best_dir[1]}")
        
        num_movements += 1
        timesteps += 1

    # ----------------- PHASE 2: Rat Finding -----------------

    # store current position as only possible cell based on localization
    curr_r, curr_c = set_possible_cells.pop()

    # update info variables to be aligned with new changes and current ship state and bot position
    info['ship'][bot_r][bot_c] = 0
    info['ship'][curr_r][curr_c] = 2
    info['bot'] = (curr_r, curr_c)

    # Sync bot_r, bot_c with new position
    bot_r, bot_c = curr_r, curr_c  

    # Create Probability Map s.t. rat_prob_map[r][c] = probability that rat is at position r,c based on ping information so far
    rat_prob_map = copy.deepcopy(info['empty_ship'])
    
    # Initialize all the probabilities for open cells to be uniformly the same number
    # let's say we have x open cells. At the beginning, P(rat at r,c) = 1/x for all open cells (r,c)
    num_open_cells = 0
    for i in range(len(rat_prob_map)):
        for j in range(len(rat_prob_map)):
            if rat_prob_map[i][j] == 0:  # open
                num_open_cells += 1
            else:
                rat_prob_map[i][j] = -1    
    uniform_prob_i = 1 / num_open_cells

    # store coordinates for all open cells - useful for probability updates in ratfinding process
    open_cells = set()
    for i in range(len(rat_prob_map)):
        for j in range(len(rat_prob_map)):
            if rat_prob_map[i][j] == 0: 
                rat_prob_map[i][j] = uniform_prob_i
                open_cells.add((i,j))

    # empty path which stores the current planned path coordinates of the bot to its next destination on the ship
    path = []

    # Initial ping before the loop
    ping_result = ping(info, alpha)

    # Increment counter variables appropriately
    num_space_rat_pings += 1
    timesteps += 1

    # if we are at the rat, return all the counter variables for all actions
    if ping_result == 'Found':
        return num_blocked_cell_detects, num_space_rat_pings, num_movements, timesteps
    
    # if we are not at the rat, update rat_prob_map
    else:
        
        # rat cannot be at this position because we have not found the rat
        rat_prob_map[bot_r][bot_c] = 0

        # update probability that rat is at any other cell based on whether or not we heard a ping using conditional probability calculations described in write-up
        summed_prob = 0

        if ping_result == 'True':
            for (r,c) in open_cells:
                prob_rat = math.e ** (-alpha * ((abs(r - bot_r) + abs(c - bot_c))-1))
                rat_prob_map[r][c] *= prob_rat
                summed_prob += rat_prob_map[r][c]
            for (r,c) in open_cells:
                rat_prob_map[r][c] /= summed_prob

        elif ping_result == 'False':
            for (r,c) in open_cells:
                rat_prob_map[r][c] *= (1 - math.e ** (-alpha * ((abs(r - bot_r) + abs(c - bot_c))-1)))
                summed_prob += rat_prob_map[r][c]
            for (r,c) in open_cells:
                rat_prob_map[r][c] /= summed_prob

    # found variable that keeps track of whether or not rat has been found
    found = False

    # keep pinging and moving until we find the rat
    while not found:
        
        if visualize: visualize_rat_prob_map(rat_prob_map)
        
        # determine the coordinates of the cell where we think the rat is most likely at based on all the ping results we have done so far
        highest_rat_prob = 0
        highest_rat_prob_cell = (-1,-1)
        for i in range(len(rat_prob_map)):
            for j in range(len(rat_prob_map)):
                if rat_prob_map[i][j] > highest_rat_prob:
                    highest_rat_prob = rat_prob_map[i][j]
                    highest_rat_prob_cell = (i,j)

        # plan a shortest distance path to that cell using A*
        path = astar(info['bot'], info['empty_ship'], highest_rat_prob_cell)

        # commit to that path and move along every cell in the shortest distance path
        for new_r, new_c in path:
            
            # Move
            info['ship'][bot_r][bot_c] = 0  # Clear old position
            info['bot'] = (new_r, new_c) # Update new bot position
            info['ship'][new_r][new_c] = 2  # Set new position in ship
            bot_r, bot_c = new_r, new_c  # Update bot_r, bot_c
            
            # increment counter variables by 1 - move and timesteps
            num_movements += 1
            timesteps += 1
            
            # ping the rat and increment appropriate counter variables
            ping_result = ping(info, alpha)
            num_space_rat_pings += 1
            timesteps += 1

            # if rat is found, we are done; return counter variables
            if ping_result == 'Found':
                return num_blocked_cell_detects, num_space_rat_pings, num_movements, timesteps

            # if rat is not found, update rat_prob_map based on conditional probability approach described in write up
            else:
                summed_prob = 0
                rat_prob_map[bot_r][bot_c] = 0
                if ping_result == 'True':
                    for (r,c) in open_cells:
                        prob_rat = math.e ** (-alpha * ((abs(r - bot_r) + abs(c - bot_c))-1))
                        rat_prob_map[r][c] *= prob_rat
                        summed_prob += rat_prob_map[r][c]
                    for (r,c) in open_cells:
                        rat_prob_map[r][c] /= summed_prob
                elif ping_result == 'False':
                    for (r,c) in open_cells:
                        rat_prob_map[r][c] *= (1 - math.e ** (-alpha * ((abs(r - bot_r) + abs(c - bot_c))-1)))
                        summed_prob += rat_prob_map[r][c]
                    for (r,c) in open_cells:
                        rat_prob_map[r][c] /= summed_prob
            
            if visualize:
                visualize_ship(ship, path, title = "Planned Path")
                visualize_rat_prob_map(rat_prob_map, None, title = f"Rat_prob_map Timestep = {timesteps}")

# Bot 1 Implementation here

def init_ship(dimension):
    d = dimension - 2
    ship = [[1] * d for _ in range(d)] 
    to_open = random.sample(range(1,d-1), 2)
    row, col = to_open
    ship[row][col] = 0

    single_neighbor = set()
    closed = set()
    
    for dr, dc in directions:
        r = row + dr
        c = col + dc
        if 0 <= r < d and 0 <= c < d:
            single_neighbor.add((r,c))

    while single_neighbor:
        chosen_coordinate = random.choice(list(single_neighbor))
        single_neighbor.remove(chosen_coordinate)
        row, col = chosen_coordinate 
        ship[row][col] = 0
        for dr,dc in directions:
            r = row + dr
            c = col + dc
            if 0 <= r < d and 0 <= c < d and ship[r][c] == 1 and (r,c) not in closed:
                if (r,c) in single_neighbor:
                    single_neighbor.remove((r,c))
                    closed.add((r,c))
                else:
                    single_neighbor.add((r,c))
    
    deadends = dict()
    for r in range(d):
        for c in range(d):
            if ship[r][c] == 0:
                open_n_count = 0
                closed_neighbors = []
                for dr,dc in directions:
                    nr,nc = r + dr, c + dc
                    if 0 <= nr < d and 0 <= nc < d:
                        if ship[nr][nc] == 0:
                            open_n_count += 1
                        elif ship[nr][nc] == 1:
                            closed_neighbors.append((nr,nc))
                if open_n_count == 1:
                    deadends[(r,c)] = closed_neighbors

    for i in range(len(deadends)//2):
        list_closed_neighbors = deadends.pop(random.choice(list(deadends.keys())))
        r,c = random.choice(list_closed_neighbors)
        ship[r][c] = 0

    ship.insert(0,[1] * dimension)
    ship.append([1] * dimension)
    for i in range(1,dimension-1):
        row = ship[i]
        new_row = [1] + row + [1]
        ship[i] = new_row
    
    open_cells = set()
    for r in range(dimension):
        for c in range(dimension):
            if ship[r][c] == 0:
                open_cells.add((r,c))
    
    # print("num open cells", len(open_cells))

    empty_ship = copy.deepcopy(ship)
    bot_r,bot_c = random.choice(list(open_cells))
    open_cells.remove((bot_r,bot_c))
    rat_r,rat_c = random.choice(list(open_cells))
    open_cells.remove((rat_r,rat_c))

    ship[bot_r][bot_c] = 2
    ship[rat_r][rat_c] = 3

    info = dict()
    info['ship'] = ship
    info['empty_ship'] = empty_ship
    info['bot'] = (bot_r, bot_c)
    info['rat'] = (rat_r, rat_c)
    return info

def astar(start, map, button):
    def heuristic(cell1):
        return abs(cell1[0] - button[0]) + abs(cell1[1]-button[1]) 
    
    d = len(map)
    fringe = []
    heapq.heappush(fringe, (heuristic(start),start))
    total_costs = dict()
    total_costs[start] = 0
    prev = dict()
    prev[start] = None

    while fringe:
        curr = heapq.heappop(fringe)
        if curr[1] == button:
            curr_p = curr[1]
            path = deque()
            while curr_p != None:
                path.appendleft(curr_p)
                curr_p = prev[curr_p]
            return list(path)
        
        r,c = curr[1]
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            child = (nr,nc)
            if 0 <= nr < d and 0 <= nc < d and (map[nr][nc] != 1 and map[nr][nc] != -1):
                cost = total_costs[curr[1]] + 1
                est_total_cost = cost + heuristic(child)
                if child not in total_costs:
                    prev[child] = curr[1]
                    total_costs[child] = cost
                    heapq.heappush(fringe, (est_total_cost, child))
    return []        



def main():
    random.seed(20)

    og_info = init_ship(30)

    info_1 = copy.deepcopy(og_info)

    print("BOT", info_1['bot'])
    print("RAT", info_1['rat'])

    visualize_ship(og_info['ship'], None)
    
    num_blocked_cell_detects, num_space_rat_pings, num_movements, timesteps = bot1(info_1, visualize=False, alpha=0.1)
    print("BOT 1 PEFORMANCE:")
    print("num_blocked_cell_detects", num_blocked_cell_detects),
    print("num_space_rat_pings", num_space_rat_pings)
    print("num_movements", num_movements)
    print("time_steps", timesteps)

    info_2 = copy.deepcopy(og_info)

    num_blocked_cell_detects, num_space_rat_pings, num_movements, timesteps = bot3(info_2, visualize=True, alpha=0.02)
    print("BOT 2 PEFORMANCE:")
    print("num_blocked_cell_detects", num_blocked_cell_detects),
    print("num_space_rat_pings", num_space_rat_pings)
    print("num_movements", num_movements)
    print("time_steps", timesteps)

    # info_2 = copy.deepcopy(og_info)

    # print("\nBOT 2 PERFORMANCE, 0.01")
    # num_blocked_cell_detects, num_space_rat_pings, num_movements, timesteps = bot2(info_2, visualize=True, alpha=0.1)
    # print("num_blocked_cell_detects", num_blocked_cell_detects),
    # print("num_space_rat_pings", num_space_rat_pings)
    # print("num_movements", num_movements)
    # print("time_steps", timesteps)
        

if __name__ == "__main__":
    main()



def bot_old(info, visualize, alpha):
    # Get initial bot position and ship configuration
    bot_r, bot_c = info['bot']
    ship = info['empty_ship']
    d = len(info['ship'])  # dimension of ship grid

    # Initialize current bot position
    curr_r, curr_c = bot_r, bot_c

    # Generate helper variables
    neighbor_map, blocked_neighbors = create_neighbor_map(ship)
    
    # Initialize set of possible cells
    set_possible_cells = {(r, c) for r in range(d) for c in range(d) if ship[r][c] == 0}

    # Initialize counters
    num_movements = 0
    num_blocked_cell_detects = 0
    num_space_rat_pings = 0
    timesteps = 0

    # ----------------- PHASE 1: Localization -----------------
    directions = [(0, 1), (0, -1), (-1, 0), (1, 0)]
    while len(set_possible_cells) > 1:
        # Sense blocked neighbors
        num_curr_blocked_ns = neighbor_map[curr_r][curr_c]
        possible_cells = { (r, c) for r, c in set_possible_cells if neighbor_map[r][c] == num_curr_blocked_ns }
        num_blocked_cell_detects += 1
        timesteps += 1

        # Compute direction consequences
        direction_c = {(0, 1): set(), (0, -1): set(), (-1, 0): set(), (1, 0): set()}
        for pcr, pcc in possible_cells:
            for dr, dc in directions:
                nr, nc = pcr + dr, pcc + dc
                if ship[nr][nc] != 0:
                    direction_c[(dr, dc)].add((pcr, pcc))
        
        # Randomly choose direction and move
        best_dir = random.choice(directions)
        nr, nc = curr_r + best_dir[0], curr_c + best_dir[1]
        num_movements += 1
        timesteps += 1

        if ship[nr][nc] == 0:  # Successful move
            set_possible_cells = set(possible_cells).difference(direction_c[best_dir])
            curr_r, curr_c = nr, nc
            new_set_possible_cells = {(r + best_dir[0], c + best_dir[1]) for r, c in set_possible_cells}
            set_possible_cells = copy.deepcopy(new_set_possible_cells)
            info['bot'] = (curr_r, curr_c)
            info['ship'][bot_r][bot_c] = 0
            info['ship'][curr_r][curr_c] = 2
            bot_r, bot_c = curr_r, curr_c
        else:  # Unsuccessful move
            set_possible_cells = direction_c[best_dir]

    # ----------------- PHASE 2: Rat Finding -----------------
    curr_r, curr_c = set_possible_cells.pop()
    info['ship'][bot_r][bot_c] = 0
    info['ship'][curr_r][curr_c] = 2
    info['bot'] = (curr_r, curr_c)
    bot_r, bot_c = curr_r, curr_c

    # Initialize rat probability map
    rat_prob_map = copy.deepcopy(info['empty_ship'])
    num_open_cells = sum(1 for r in range(d) for c in range(d) if ship[r][c] == 0)
    uniform_prob_i = 1 / num_open_cells
    open_cells = {(r, c) for r in range(d) for c in range(d) if ship[r][c] == 0}
    for r, c in open_cells:
        rat_prob_map[r][c] = uniform_prob_i
    for r in range(d):
        for c in range(d):
            if ship[r][c] != 0:
                rat_prob_map[r][c] = -1

    # Initial ping
    ping_result = ping(info, alpha)
    num_space_rat_pings += 1
    timesteps += 1

    if ping_result == 'Found':
        return num_blocked_cell_detects, num_space_rat_pings, num_movements, timesteps

    # Update rat_prob_map
    rat_prob_map[bot_r][bot_c] = 0
    summed_prob = 0
    if ping_result == 'True':
        for r, c in open_cells:
            prob_rat = math.e ** (-alpha * (abs(r - bot_r) + abs(c - bot_c) - 1))
            rat_prob_map[r][c] *= prob_rat
            summed_prob += rat_prob_map[r][c]
    elif ping_result == 'False':
        for r, c in open_cells:
            prob_rat = 1 - math.e ** (-alpha * (abs(r - bot_r) + abs(c - bot_c) - 1))
            rat_prob_map[r][c] *= prob_rat
            summed_prob += rat_prob_map[r][c]
    for r, c in open_cells:
        rat_prob_map[r][c] /= summed_prob

    # Rat finding loop
    found = False
    while not found:
        if visualize:
            visualize_rat_prob_map(rat_prob_map)

        # Compute expected rat position
        E_r, E_c = 0, 0
        total_prob = 0
        for r, c in open_cells:
            prob = rat_prob_map[r][c]
            E_r += r * prob
            E_c += c * prob
            total_prob += prob
        if total_prob > 0:
            E_r /= total_prob
            E_c /= total_prob

        # Find nearest open cell to expected position
        target_r, target_c = min(open_cells, key=lambda pos: (pos[0] - E_r)**2 + (pos[1] - E_c)**2)

        # Plan path to target using A*
        path = astar(info['bot'], info['empty_ship'], (target_r, target_c))

        # Move along path
        for new_r, new_c in path:
            info['ship'][bot_r][bot_c] = 0
            info['bot'] = (new_r, new_c)
            info['ship'][new_r][new_c] = 2
            bot_r, bot_c = new_r, new_c
            num_movements += 1
            timesteps += 1

            # Ping and check
            ping_result = ping(info, alpha)
            num_space_rat_pings += 1
            timesteps += 1

            if ping_result == 'Found':
                return num_blocked_cell_detects, num_space_rat_pings, num_movements, timesteps

            # Update rat_prob_map
            rat_prob_map[bot_r][bot_c] = 0
            summed_prob = 0
            if ping_result == 'True':
                for r, c in open_cells:
                    prob_rat = math.e ** (-alpha * (abs(r - bot_r) + abs(c - bot_c) - 1))
                    rat_prob_map[r][c] *= prob_rat
                    summed_prob += rat_prob_map[r][c]
            elif ping_result == 'False':
                for r, c in open_cells:
                    prob_rat = 1 - math.e ** (-alpha * (abs(r - bot_r) + abs(c - bot_c) - 1))
                    rat_prob_map[r][c] *= prob_rat
                    summed_prob += rat_prob_map[r][c]
            for r, c in open_cells:
                rat_prob_map[r][c] /= summed_prob

            if visualize:
                visualize_ship(ship, path, title="Planned Path")
                visualize_rat_prob_map(rat_prob_map, None, title=f"Rat_prob_map Timestep = {timesteps}")
