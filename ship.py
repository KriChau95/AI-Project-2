# ship.py represents all the information to simulate the space vessel and contains bot functions for each bot

# Importing libraries for randomness, data structures, and data visualization
import random
import heapq
from collections import deque, defaultdict
import copy
import math
from visualize import *

# setting up global variables that are used for adjacency in searches
global directions
directions = [(0,1), (0,-1), (1,0), (-1,0)] # array to store adjacent directions needed during various traversal
global diagonal_directions
diagonal_directions = [(0,1), (0,-1), (1,0), (-1,0), (-1,1), (-1,-1), (1,1), (1,-1)]

# function to initialize ship by creating maze structure and randomly placing bot and rat
def init_ship(dimension):

    # 0 = open cell
    # 1 = closed cell
    # 2 = bot
    # 3 = rat

    d = dimension - 2

    # initialize ship to size dimension
    ship = [[1] * d for _ in range(d)] 

    # open up a random cell on the interior
    to_open = random.sample(range(1,d-1), 2)
    row, col = to_open
    ship[row][col] = 0

    single_neighbor = set() # stores all cells' blocked coordinates that have exactly 1 open neighbor
    closed = set() # stores cells that have no chance for being blocked coordinates with exactly 1 open neighbor

    # initialize single_neighbor set based on first open cell
    for dr, dc in directions:
        r = row + dr
        c = col + dc
        if 0 <= r < d and 0 <= c < d:
            single_neighbor.add((r,c))

    # Iteratively opening up cells to create maze structure
    while single_neighbor:

        chosen_coordinate = random.choice(list(single_neighbor)) # choose cell randomly
        single_neighbor.remove(chosen_coordinate) # once cell is open, it can no longer be a blocked cell
        
        row, col = chosen_coordinate 
        ship[row][col] = 0 # open it up
        
        # determine which cells are new candidates to be single neighbors and add cells that have already been dealt with to a closed set
        for dr,dc in directions:
            r = row + dr
            c = col + dc
            if 0 <= r < d and 0 <= c < d and ship[r][c] == 1 and (r,c) not in closed:
                if (r,c) in single_neighbor:
                    single_neighbor.remove((r,c))
                    closed.add((r,c))
                else:
                    single_neighbor.add((r,c))
    
    # Identifying and handling deadend cells
    
    deadends = dict()

    # deadends = open cells with exactly 1 open neighbor
    # deadends dictionary:
    # key: (r,c) s.t. (r,c) is an open cell with exactly 1 open neighbor
    # value: list of (r,c) tuples that represent key's closed neighbors
    
    for r in range(d):
        for c in range(d):
            if ship[r][c] == 0: # open cell
                open_n_count = 0
                closed_neighbors = []
                for dr,dc in directions:
                    nr,nc = r + dr, c + dc
                    if 0 <= nr < d and 0 <= nc < d:
                        if ship[nr][nc] == 0: # open neighbor
                            open_n_count += 1
                        elif ship[nr][nc] == 1:
                            closed_neighbors.append((nr,nc))
                if open_n_count == 1:
                    deadends[(r,c)] = closed_neighbors

    # for ~ 1/2 of deadend cells, pick 1 of their closed neighbors at random and open it
    for i in range(len(deadends)//2):
        list_closed_neighbors = deadends.pop(random.choice(list(deadends.keys()))) # retrieve a random deadend cell's list of closed neighbors
        r,c = random.choice(list_closed_neighbors) # choose a random closed neighbor
        ship[r][c] = 0 # open it

    # ensure border is closed
    ship.insert(0,[1] * dimension)
    ship.append([1] * dimension)
    for i in range(1,dimension-1):
        row = ship[i]
        new_row = [1] + row + [1]
        ship[i] = new_row
    
    # determine remaining open cells
    open_cells = set()
    for r in range(dimension):
        for c in range(dimension):
            if ship[r][c] == 0:
                open_cells.add((r,c))

    # Condense all the information created within this function into a hashmap and return the hashmap

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

# Baseline Bot 1 Stationary Case
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

        # if we hear a ping at this position, based on alpha, determine how likely it is that rat is at every cell given we heard a ping here
        # update rat probability map accordingly
        if ping_result == 'True':
            for (r,c) in open_cells:
                prob_rat = math.e ** (-alpha * ((abs(r - bot_r) + abs(c - bot_c))-1))
                rat_prob_map[r][c] *= prob_rat
                summed_prob += rat_prob_map[r][c]
            for (r,c) in open_cells:
                rat_prob_map[r][c] /= summed_prob

        # if do not we hear a ping at this position, based on alpha, determine how likely it is that rat is at every cell given we did not hear a ping here
        # update rat probability map accordingly
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

# Improved Version of Baseline Bot Stationary Case
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
    
    # Initialize result variables that provide insight to rat-finding process
    num_movements = 0
    num_blocked_cell_detects = 0
    num_space_rat_pings = 0
    timesteps = 0  # Tracks total actions (sense, ping, move)

    # ----------------- PHASE 1: Localization -----------------
    # Repeat until only one possible bot location remains
    while len(possible_cells) > 1:
        
        # Sense blocked neighbors (alternating action 1)
        num_curr_blocked_ns = neighbor_map[curr_r][curr_c]
        
        # narrow down possible cells to only those that match the current number of blocked neighbors sensed
        possible_cells = set()
        for cellr, cellc in set_possible_cells:
            if neighbor_map[cellr][cellc] == num_curr_blocked_ns:
                possible_cells.add((cellr, cellc))
        
        # increment appropriate counters
        num_blocked_cell_detects += 1
        timesteps += 1

        # Determine the most commonly open direction and attempt to move (alternating action 2)
        direction_c = {(0,1): set(), (0,-1): set(), (-1,0): set(), (1,0): set()}

        for pcr, pcc in possible_cells:
            for dr, dc in directions:
                nr = pcr + dr
                nc = pcc + dc
                if ship[nr][nc] != 0:
                    direction_c[(dr,dc)].add((pcr,pcc))

        # randomly choose a direction to move in
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
        
        num_movements += 1
        timesteps += 1

    # ----------------- PHASE 2: Rat Finding -----------------
    
    # set current position to determined position
    curr_r, curr_c = set_possible_cells.pop()
    info['ship'][bot_r][bot_c] = 0
    info['ship'][curr_r][curr_c] = 2
    info['bot'] = (curr_r, curr_c)
    bot_r, bot_c = curr_r, curr_c  # Sync bot_r, bot_c with new position
    
    def dfs_explore(info, start, ship):
        stack = [(start, [start])]  # (current position, path so far)
        visited = set([start])
        num_movements = 0
        num_space_rat_pings = 0
        timesteps = 0
        d = len(ship)
        directions = [(0, 1), (0, -1), (-1, 0), (1, 0)]  # right, left, up, down

        while stack:
            (r, c), path = stack[-1]  # current position
            
            # Ping at current position if not yet pinged
            if (r, c) not in visited:
                # Move bot to current position (if not already there)
                if info['bot'] != (r, c):
                    info['ship'][info['bot'][0]][info['bot'][1]] = 0  # Clear old position
                    info['bot'] = (r, c)
                    info['ship'][r][c] = 2  # Update new position
                    num_movements += 1
                    timesteps += 1
                    if visualize: visualize_side_by_side(info['empty_ship'], info['ship'], path, info['bot'], f"Timestep {timesteps}")
                
                # Ping to check if rat is here
                ping_result = ping(info, 0)  # alpha = 0
                num_space_rat_pings += 1
                timesteps += 1
                visited.add((r, c))  # Mark as visited after pinging
                if visualize: visualize_side_by_side(info['empty_ship'], info['ship'], path, info['bot'], f"Timestep {timesteps}")

                if ping_result == "Found":
                    return num_movements, num_space_rat_pings, timesteps, path

            # Find unvisited neighbors
            unvisited_neighbors = []
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if (0 <= nr < d and 0 <= nc < d and 
                    ship[nr][nc] == 0 and (nr, nc) not in visited):
                    unvisited_neighbors.append((nr, nc))

            if unvisited_neighbors:
                # Choose the first unvisited neighbor
                next_r, next_c = unvisited_neighbors[0]
                new_path = path + [(next_r, next_c)]
                stack.append(((next_r, next_c), new_path))
                
                # Move to the neighbor
                info['ship'][info['bot'][0]][info['bot'][1]] = 0  # Clear old position
                info['bot'] = (next_r, next_c)
                info['ship'][next_r][next_c] = 2  # Update new position
                num_movements += 1
                timesteps += 1
                if visualize: visualize_side_by_side(info['empty_ship'], info['ship'], new_path, info['bot'], f"Timestep {timesteps}")
            else:
                # No unvisited neighbors, backtrack
                stack.pop()  # Remove current position from stack
                if stack:
                    # Move back to the previous position in the stack
                    prev_r, prev_c = stack[-1][0]
                    info['ship'][info['bot'][0]][info['bot'][1]] = 0  # Clear current position
                    info['bot'] = (prev_r, prev_c)
                    info['ship'][prev_r][prev_c] = 2  # Update to previous position
                    num_movements += 1
                    timesteps += 1
                    if visualize: visualize_side_by_side(info['empty_ship'], info['ship'], stack[-1][1], info['bot'], f"Timestep {timesteps}")

        # If we exit the loop without finding the rat, return current counts
        return num_movements, num_space_rat_pings, timesteps, path

    # Integration into the main bot2 function
    if alpha == 0:
        # Special case: Use DFS exploration when alpha = 0
        dfs_moves, dfs_pings, dfs_timesteps, path = dfs_explore(info, (curr_r, curr_c), info['ship'])
        num_movements += dfs_moves
        num_space_rat_pings += dfs_pings
        timesteps += dfs_timesteps
        if visualize:
            visualize_ship(info['ship'], path, title="DFS Path when alpha = 0")
        return num_blocked_cell_detects, num_space_rat_pings, num_movements, timesteps
    
    # Initialize all the probabilities for open cells to be uniformly the same number
    # let's say we have x open cells. At the beginning, P(rat at r,c) = 1/x for all open cells (r,c)
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

        # Update rat probability knowledge base based on if we heard a ping or not
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
        
        # determine most likely position of rat based on probability knowledge base
        highest_rat_prob = 0
        highest_rat_prob_cell = (-1,-1)
        for i in range(len(rat_prob_map)):
            for j in range(len(rat_prob_map)):
                temp = rat_prob_map[i][j] 
                if temp > highest_rat_prob:
                    highest_rat_prob = temp
                    highest_rat_prob_cell = (i,j)
        
        # Plan path to position of highest rat probability using alternate A* that promotes visiting cells with high probability
        path = astar_with_heuristic(info['bot'], rat_prob_map, info['empty_ship'], highest_rat_prob_cell)
        
        # max number of steps to move in a path before recalculating and determining if knowledge base tells us to recalibrate to a different path
        steps_towards_before_recalc = 15

        # move along planned path until either path is finished or max number of steps is reached
        for new_r, new_c in path[:steps_towards_before_recalc]:
            
            # Move
            info['ship'][bot_r][bot_c] = 0  # Clear old position
            info['bot'] = (new_r, new_c)
            info['ship'][new_r][new_c] = 2  # Set new position
            bot_r, bot_c = new_r, new_c  # Update bot_r, bot_c
            
            # Appropriately increment counter variables
            num_movements += 1
            timesteps += 1

            # If we know that the rat is not here, we will not ping, and instead move to the next step in the path
            if rat_prob_map[bot_r][bot_c] == 0:
                continue
            
            # Ping if this cell has a possibility of having the rat
            ping_result = ping(info, alpha)

            # Increment appropriate counters
            num_space_rat_pings += 1
            timesteps += 1

            # Update probabilistic knowledge base depending on whether or not we heard a ping
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
                
            if visualize:
                visualize_ship(ship, path, title = "Planned Path")
                visualize_rat_prob_map(rat_prob_map, None, title = f"Rat_prob_map Timestep = {timesteps}")

# A* algorithm to find shortest path from any start position to any end position 
def astar(start, map, end):
    def heuristic(cell1):
        return abs(cell1[0] - end[0]) + abs(cell1[1]-end[1]) 
    
    d = len(map)
    fringe = []
    heapq.heappush(fringe, (heuristic(start),start))
    total_costs = dict()
    total_costs[start] = 0
    prev = dict()
    prev[start] = None

    while fringe:
        curr = heapq.heappop(fringe)
        if curr[1] == end:
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

# Modified version of A* that promotes visiting cells with high rat probability even though they may be longer by reducing cost at a cell in a way proportional to its rat probability
def astar_with_heuristic(start, rat_prob_map, map, end):
    def heuristic(cell1):
        return abs(cell1[0] - end[0]) + abs(cell1[1] - end[1])  

    d = len(map)
    fringe = []
    heapq.heappush(fringe, (heuristic(start), start))
    total_costs = dict()
    total_costs[start] = 0
    prev = dict()
    prev[start] = None

    while fringe:
        curr = heapq.heappop(fringe)
        if curr[1] == end:
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

                # promoting a path that visits cells with high rat probabilities
                cost -= (1000 * rat_prob_map[nr][nc])
                cost = max(cost, 0)

                est_total_cost = cost + heuristic(child)
                if child not in total_costs:
                    prev[child] = curr[1]
                    total_costs[child] = cost
                    heapq.heappush(fringe, (est_total_cost, child))
    
    return []    

# Main for testing
def main():
    random.seed(21)

    og_info = init_ship(30)

    info_1 = copy.deepcopy(og_info)

    print("BOT", info_1['bot'])
    print("RAT", info_1['rat'])

    # visualize_ship(og_info['ship'], None)
    
    num_blocked_cell_detects, num_space_rat_pings, num_movements, timesteps = bot1(info_1, visualize=False, alpha=0.00)
    print("BOT 1 PEFORMANCE:")
    print("num_blocked_cell_detects", num_blocked_cell_detects),
    print("num_space_rat_pings", num_space_rat_pings)
    print("num_movements", num_movements)
    print("time_steps", timesteps)

    info_2 = copy.deepcopy(og_info)

    num_blocked_cell_detects, num_space_rat_pings, num_movements, timesteps = bot2(info_2, visualize=True, alpha=0.00)
    print("BOT 2 PEFORMANCE:")
    print("num_blocked_cell_detects", num_blocked_cell_detects),
    print("num_space_rat_pings", num_space_rat_pings)
    print("num_movements", num_movements)
    print("time_steps", timesteps)

if __name__ == "__main__":
    main()