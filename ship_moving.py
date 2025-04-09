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


# New helper function to compute the transition matrix (dictionary of dictionaries) to account for probabilities of rat movement
def compute_transition_matrix(ship, open_cells):
    
    d = len(ship)
    
    # create a dictionary of dictionaries to store the probability that the rat moves from one state to the next
    T = defaultdict(lambda: defaultdict(float))

    # iterate through all open cells
    for r, c in open_cells:
        
        # for each open cell, store all of its adjacent open neighbors to a neighbors list
        neighbors = []
        for dr, dc in directions:
            if 0 <= r + dr < d and 0 <= c + dc < d and ship [r + dr][c + dc] == 0:
                neighbors.append((r + dr, c + dc))

        # calculate and store the probability that a rat can move from this open cell to each specific neighbor cell as a value
        # dictionary - key = cell, value = dictionary of neighboring cells s.t. key = neighboring cel, value = probability of moving to that neighboring cell
        prob = 1 / len(neighbors)
        
        for nr, nc in neighbors:
            T[(r, c)][(nr, nc)] = prob
    
    # return the information stored in T as well as the open cells
    return T

# Baseline Bot 1 that has overall algorithmic approach similar to Baseline bot 1 in stationary case
# Main difference is that probabilistic knowledge base is modified to appropriately account for moving rat
def bot1_2(info, visualize, alpha):

    # Get initial bot position and ship configuration
    bot_r, bot_c = info['bot']
    rat_r, rat_c = info['rat']
    ship = info['empty_ship']
    d = len(info['ship'])

    # store all open cell coordinates in a list
    open_cells = []
    for r in range(d):
        for c in range(d):
            if ship[r][c] == 0:
                open_cells.append((r,c))

    # Initialize current positions
    curr_r, curr_c = bot_r, bot_c
    curr_rat_r, curr_rat_c = rat_r, rat_c

    # Generate helper variables
    neighbor_map, blocked_neighbors = create_neighbor_map(ship)
    set_possible_cells = {(r, c) for r in range(d) for c in range(d) if ship[r][c] == 0}

    # Initialize result variables
    num_movements = 0
    num_blocked_cell_detects = 0
    num_space_rat_pings = 0
    timesteps = 0

    # ----------------- PHASE 1: Localization -----------------
    while len(set_possible_cells) > 1:
       
        # Sense blocked neighbors
        num_curr_blocked_ns = neighbor_map[curr_r][curr_c]
        possible_cells = {cell for cell in set_possible_cells if neighbor_map[cell[0]][cell[1]] == num_curr_blocked_ns}
        num_blocked_cell_detects += 1
        timesteps += 1

        # Rat moves
        rat_move_choices = [(dr, dc) for dr, dc in directions if ship[curr_rat_r + dr][curr_rat_c + dc] == 0]
        info['ship'][curr_rat_r][curr_rat_c] = 0
        actual_rat_move = random.choice(rat_move_choices)
        curr_rat_r += actual_rat_move[0]
        curr_rat_c += actual_rat_move[1]
        info['ship'][curr_rat_r][curr_rat_c] = 3
        info['rat'] = (curr_rat_r, curr_rat_c)

        # Determine possible moves
        direction_c = {(0, 1): set(), (0, -1): set(), (-1, 0): set(), (1, 0): set()}
        for pcr, pcc in possible_cells:
            for dr, dc in directions:
                nr, nc = pcr + dr, pcc + dc
                if ship[nr][nc] != 0:
                    direction_c[(dr, dc)].add((pcr, pcc))

        # Randomly choose a direction and move
        best_dir = random.choice(directions)
        nr, nc = curr_r + best_dir[0], curr_c + best_dir[1]

        if ship[nr][nc] == 0:  # Successful move
            set_possible_cells = set(possible_cells).difference(direction_c[best_dir])
            curr_r, curr_c = nr, nc
            new_set_possible_cells = {(r + best_dir[0], c + best_dir[1]) for r, c in set_possible_cells}
            set_possible_cells = new_set_possible_cells
            info['bot'] = (curr_r, curr_c)
            info['ship'][bot_r][bot_c] = 0
            info['ship'][curr_r][curr_c] = 2
            bot_r, bot_c = curr_r, curr_c
        else:  # Unsuccessful move
            set_possible_cells = direction_c[best_dir]

        num_movements += 1
        timesteps += 1

        # Rat moves again
        rat_move_choices = [(dr, dc) for dr, dc in directions if ship[curr_rat_r + dr][curr_rat_c + dc] == 0]
        info['ship'][curr_rat_r][curr_rat_c] = 0
        actual_rat_move = random.choice(rat_move_choices)
        curr_rat_r += actual_rat_move[0]
        curr_rat_c += actual_rat_move[1]
        info['ship'][curr_rat_r][curr_rat_c] = 3
        info['rat'] = (curr_rat_r, curr_rat_c)

    # ----------------- PHASE 2: Rat Finding -----------------
    
    curr_r, curr_c = set_possible_cells.pop()
    info['ship'][bot_r][bot_c] = 0
    info['ship'][curr_r][curr_c] = 2
    info['bot'] = (curr_r, curr_c)
    bot_r, bot_c = curr_r, curr_c

    # Initialize probability map and transition matrix
    rat_prob_map = [[-1 if ship[r][c] == 1 else 0 for c in range(d)] for r in range(d)]
    T = compute_transition_matrix(ship, open_cells)
    num_open_cells = len(open_cells)
    uniform_prob_i = 1 / num_open_cells
    for r, c in open_cells:
        rat_prob_map[r][c] = uniform_prob_i

    path = []
    found = False

    # Initial ping
    ping_result = ping(info, alpha)
    num_space_rat_pings += 1
    timesteps += 1
    
    # rat moves because a timestep (ping) occurred
    rat_move_choices = [(dr, dc) for dr, dc in directions if ship[curr_rat_r + dr][curr_rat_c + dc] == 0]
    info['ship'][curr_rat_r][curr_rat_c] = 0
    actual_rat_move = random.choice(rat_move_choices)
    curr_rat_r += actual_rat_move[0]
    curr_rat_c += actual_rat_move[1]
    info['ship'][curr_rat_r][curr_rat_c] = 3
    info['rat'] = (curr_rat_r, curr_rat_c)

    if ping_result == 'Found':
        return num_blocked_cell_detects, num_space_rat_pings, num_movements, timesteps

    # Update probabilities based on initial ping
    rat_prob_map[bot_r][bot_c] = 0
    summed_prob = 0
    if ping_result == 'True':
        for r, c in open_cells:
            prob_rat = math.e ** (-alpha * (abs(r - bot_r) + abs(c - bot_c) - 1))
            rat_prob_map[r][c] *= prob_rat
            summed_prob += rat_prob_map[r][c]
    else:  # 'False'
        for r, c in open_cells:
            rat_prob_map[r][c] *= (1 - math.e ** (-alpha * (abs(r - bot_r) + abs(c - bot_c) - 1)))
            summed_prob += rat_prob_map[r][c]
    for r, c in open_cells:
        rat_prob_map[r][c] /= summed_prob

    # Update probabilities based on rat movement
    new_rat_prob_map = [[-1 if ship[r][c] == 1 else 0 for c in range(d)] for r in range(d)]
    for r, c in open_cells:
        for nr, nc in T[(r, c)]:
            new_rat_prob_map[nr][nc] += rat_prob_map[r][c] * T[(r, c)][(nr, nc)]
    rat_prob_map = new_rat_prob_map

    # Main rat-finding loop
    while not found:
        if visualize:
            visualize_side_by_side(rat_prob_map, info['ship'], path, info['bot'], title=f"Timesteps {timesteps}")

        # Find cell with highest probability
        highest_rat_prob = 0
        highest_rat_prob_cell = (-1, -1)
        for r, c in open_cells:
            if rat_prob_map[r][c] > highest_rat_prob:
                highest_rat_prob = rat_prob_map[r][c]
                highest_rat_prob_cell = (r, c)

        # Plan shortest path using A*
        path = astar(info['bot'], info['empty_ship'], highest_rat_prob_cell)

        # Move along the path
        for new_r, new_c in path:
            # Move bot
            info['ship'][bot_r][bot_c] = 0
            info['bot'] = (new_r, new_c)
            info['ship'][new_r][new_c] = 2
            bot_r, bot_c = new_r, new_c
            num_movements += 1
            timesteps += 1

            # Rat moves
            rat_move_choices = [(dr, dc) for dr, dc in directions if ship[curr_rat_r + dr][curr_rat_c + dc] == 0]
            info['ship'][curr_rat_r][curr_rat_c] = 0
            actual_rat_move = random.choice(rat_move_choices)
            curr_rat_r += actual_rat_move[0]
            curr_rat_c += actual_rat_move[1]
            info['ship'][curr_rat_r][curr_rat_c] = 3
            info['rat'] = (curr_rat_r, curr_rat_c)

            # Ping
            ping_result = ping(info, alpha)
            num_space_rat_pings += 1
            timesteps += 1

            if ping_result == 'Found':
                return num_blocked_cell_detects, num_space_rat_pings, num_movements, timesteps

            # Update probabilities based on ping
            rat_prob_map[bot_r][bot_c] = 0
            summed_prob = 0
            if ping_result == 'True':
                for r, c in open_cells:
                    prob_rat = math.e ** (-alpha * (abs(r - bot_r) + abs(c - bot_c) - 1))
                    rat_prob_map[r][c] *= prob_rat
                    summed_prob += rat_prob_map[r][c]
            else:  # 'False'
                for r, c in open_cells:
                    rat_prob_map[r][c] *= (1 - math.e ** (-alpha * (abs(r - bot_r) + abs(c - bot_c) - 1)))
                    summed_prob += rat_prob_map[r][c]
            for r, c in open_cells:
                rat_prob_map[r][c] /= summed_prob if summed_prob > 0 else 1

            # Update probabilities based on rat movement
            new_rat_prob_map = [[-1 if ship[r][c] == 1 else 0 for c in range(d)] for r in range(d)]
            for r, c in open_cells:
                for nr, nc in T[(r, c)]:
                    new_rat_prob_map[nr][nc] += rat_prob_map[r][c] * T[(r, c)][(nr, nc)]
            rat_prob_map = new_rat_prob_map

            # Rat moves again
            rat_move_choices = [(dr, dc) for dr, dc in directions if ship[curr_rat_r + dr][curr_rat_c + dc] == 0]
            info['ship'][curr_rat_r][curr_rat_c] = 0
            actual_rat_move = random.choice(rat_move_choices)
            curr_rat_r += actual_rat_move[0]
            curr_rat_c += actual_rat_move[1]
            info['ship'][curr_rat_r][curr_rat_c] = 3
            info['rat'] = (curr_rat_r, curr_rat_c)

            if visualize:
                visualize_side_by_side(rat_prob_map, info['ship'], path, info['bot'], title=f"Timesteps {timesteps}")

    return num_blocked_cell_detects, num_space_rat_pings, num_movements, timesteps

# Original Bot 2 Ideas adjusted to simulate Moving Rat and corresponding adjusted probabilistic knowledge base updates
def bot2_2(info, visualize, alpha):
    
    # Get initial bot position and ship configuration
    bot_r, bot_c = info['bot']
    rat_r, rat_c = info['rat']
    ship = info['empty_ship']
    d = len(info['ship'])

    # store all open cell coordinates in a list
    open_cells = []
    for r in range(d):
        for c in range(d):
            if ship[r][c] == 0:
                open_cells.append((r,c))

    # Initialize current positions
    curr_r, curr_c = bot_r, bot_c
    curr_rat_r, curr_rat_c = rat_r, rat_c

    # Generate helper variables
    neighbor_map, blocked_neighbors = create_neighbor_map(ship)
    set_possible_cells = {(r, c) for r in range(d) for c in range(d) if ship[r][c] == 0}

    # Initialize result variables
    num_movements = 0
    num_blocked_cell_detects = 0
    num_space_rat_pings = 0
    timesteps = 0

    # ----------------- PHASE 1: Localization -----------------
    while len(set_possible_cells) > 1:
        # Sense blocked neighbors
        num_curr_blocked_ns = neighbor_map[curr_r][curr_c]
        possible_cells = {cell for cell in set_possible_cells if neighbor_map[cell[0]][cell[1]] == num_curr_blocked_ns}
        num_blocked_cell_detects += 1
        timesteps += 1

        # Rat moves
        rat_move_choices = [(dr, dc) for dr, dc in directions if ship[curr_rat_r + dr][curr_rat_c + dc] == 0]
        info['ship'][curr_rat_r][curr_rat_c] = 0
        actual_rat_move = random.choice(rat_move_choices)
        curr_rat_r += actual_rat_move[0]
        curr_rat_c += actual_rat_move[1]
        info['ship'][curr_rat_r][curr_rat_c] = 3
        info['rat'] = (curr_rat_r, curr_rat_c)

        # Determine possible moves
        direction_c = {(0, 1): set(), (0, -1): set(), (-1, 0): set(), (1, 0): set()}
        for pcr, pcc in possible_cells:
            for dr, dc in directions:
                nr, nc = pcr + dr, pcc + dc
                if ship[nr][nc] != 0:
                    direction_c[(dr, dc)].add((pcr, pcc))

        # Randomly choose a direction and move
        best_dir = random.choice(directions)
        nr, nc = curr_r + best_dir[0], curr_c + best_dir[1]

        if ship[nr][nc] == 0:  # Successful move
            set_possible_cells = set(possible_cells).difference(direction_c[best_dir])
            curr_r, curr_c = nr, nc
            new_set_possible_cells = {(r + best_dir[0], c + best_dir[1]) for r, c in set_possible_cells}
            set_possible_cells = new_set_possible_cells
            info['bot'] = (curr_r, curr_c)
            info['ship'][bot_r][bot_c] = 0
            info['ship'][curr_r][curr_c] = 2
            bot_r, bot_c = curr_r, curr_c
        else:  # Unsuccessful move
            set_possible_cells = direction_c[best_dir]

        num_movements += 1
        timesteps += 1

        # Rat moves again
        rat_move_choices = [(dr, dc) for dr, dc in directions if ship[curr_rat_r + dr][curr_rat_c + dc] == 0]
        info['ship'][curr_rat_r][curr_rat_c] = 0
        actual_rat_move = random.choice(rat_move_choices)
        curr_rat_r += actual_rat_move[0]
        curr_rat_c += actual_rat_move[1]
        info['ship'][curr_rat_r][curr_rat_c] = 3
        info['rat'] = (curr_rat_r, curr_rat_c)

    # ----------------- PHASE 2: Rat Finding -----------------
    
    curr_r, curr_c = set_possible_cells.pop()
    info['ship'][bot_r][bot_c] = 0
    info['ship'][curr_r][curr_c] = 2
    info['bot'] = (curr_r, curr_c)
    bot_r, bot_c = curr_r, curr_c

    # Initialize probability map and transition matrix
    rat_prob_map = [[-1 if ship[r][c] == 1 else 0 for c in range(d)] for r in range(d)]
    T = compute_transition_matrix(ship, open_cells)
    num_open_cells = len(open_cells)
    uniform_prob_i = 1 / num_open_cells
    for r, c in open_cells:
        rat_prob_map[r][c] = uniform_prob_i

    path = []
    found = False

    # Initial ping
    ping_result = ping(info, alpha)
    num_space_rat_pings += 1
    timesteps += 1
    
    # rat moves because a timestep (ping) occurred
    rat_move_choices = [(dr, dc) for dr, dc in directions if ship[curr_rat_r + dr][curr_rat_c + dc] == 0]
    info['ship'][curr_rat_r][curr_rat_c] = 0
    actual_rat_move = random.choice(rat_move_choices)
    curr_rat_r += actual_rat_move[0]
    curr_rat_c += actual_rat_move[1]
    info['ship'][curr_rat_r][curr_rat_c] = 3
    info['rat'] = (curr_rat_r, curr_rat_c)

    if ping_result == 'Found':
        return num_blocked_cell_detects, num_space_rat_pings, num_movements, timesteps

    # Update probabilities based on initial ping
    rat_prob_map[bot_r][bot_c] = 0
    summed_prob = 0
    if ping_result == 'True':
        for r, c in open_cells:
            prob_rat = math.e ** (-alpha * (abs(r - bot_r) + abs(c - bot_c) - 1))
            rat_prob_map[r][c] *= prob_rat
            summed_prob += rat_prob_map[r][c]
    else: 
        for r, c in open_cells:
            rat_prob_map[r][c] *= (1 - math.e ** (-alpha * (abs(r - bot_r) + abs(c - bot_c) - 1)))
            summed_prob += rat_prob_map[r][c]
    for r, c in open_cells:
        rat_prob_map[r][c] /= summed_prob

    # Update probabilities based on rat movement
    new_rat_prob_map = [[-1 if ship[r][c] == 1 else 0 for c in range(d)] for r in range(d)]
    for r, c in open_cells:
        for nr, nc in T[(r, c)]:
            new_rat_prob_map[nr][nc] += rat_prob_map[r][c] * T[(r, c)][(nr, nc)]
    rat_prob_map = new_rat_prob_map

    # Main rat-finding loop
    while not found:
        
        if visualize:
            visualize_side_by_side(rat_prob_map, info['ship'], path, info['bot'], title=f"Timesteps {timesteps}")

        # Find cell with highest probability
        highest_rat_prob = 0
        highest_rat_prob_cell = (-1, -1)
        for r, c in open_cells:
            if rat_prob_map[r][c] > highest_rat_prob:
                highest_rat_prob = rat_prob_map[r][c]
                highest_rat_prob_cell = (r, c)

        # Plan shortest path using A*
        path = astar_with_heuristic(info['bot'], rat_prob_map, info['empty_ship'], highest_rat_prob_cell)

        # Move along the path
        for new_r, new_c in path[:15]:
            # Move bot
            info['ship'][bot_r][bot_c] = 0
            info['bot'] = (new_r, new_c)
            info['ship'][new_r][new_c] = 2
            bot_r, bot_c = new_r, new_c
            num_movements += 1
            timesteps += 1

            # Rat moves
            rat_move_choices = [(dr, dc) for dr, dc in directions if ship[curr_rat_r + dr][curr_rat_c + dc] == 0]
            info['ship'][curr_rat_r][curr_rat_c] = 0
            actual_rat_move = random.choice(rat_move_choices)
            curr_rat_r += actual_rat_move[0]
            curr_rat_c += actual_rat_move[1]
            info['ship'][curr_rat_r][curr_rat_c] = 3
            info['rat'] = (curr_rat_r, curr_rat_c)

            # Ping
            ping_result = ping(info, alpha)
            num_space_rat_pings += 1
            timesteps += 1

            if ping_result == 'Found':
                return num_blocked_cell_detects, num_space_rat_pings, num_movements, timesteps

            # Update probabilities based on ping
            rat_prob_map[bot_r][bot_c] = 0
            summed_prob = 0
            if ping_result == 'True':
                for r, c in open_cells:
                    prob_rat = math.e ** (-alpha * (abs(r - bot_r) + abs(c - bot_c) - 1))
                    rat_prob_map[r][c] *= prob_rat
                    summed_prob += rat_prob_map[r][c]
            else:  # 'False'
                for r, c in open_cells:
                    rat_prob_map[r][c] *= (1 - math.e ** (-alpha * (abs(r - bot_r) + abs(c - bot_c) - 1)))
                    summed_prob += rat_prob_map[r][c]
            for r, c in open_cells:
                rat_prob_map[r][c] /= summed_prob if summed_prob > 0 else 1

            # Update probabilities based on rat movement
            new_rat_prob_map = [[-1 if ship[r][c] == 1 else 0 for c in range(d)] for r in range(d)]
            for r, c in open_cells:
                for nr, nc in T[(r, c)]:
                    new_rat_prob_map[nr][nc] += rat_prob_map[r][c] * T[(r, c)][(nr, nc)]
            rat_prob_map = new_rat_prob_map

            # Rat moves again
            rat_move_choices = [(dr, dc) for dr, dc in directions if ship[curr_rat_r + dr][curr_rat_c + dc] == 0]
            info['ship'][curr_rat_r][curr_rat_c] = 0
            actual_rat_move = random.choice(rat_move_choices)
            curr_rat_r += actual_rat_move[0]
            curr_rat_c += actual_rat_move[1]
            info['ship'][curr_rat_r][curr_rat_c] = 3
            info['rat'] = (curr_rat_r, curr_rat_c)

            if visualize:
                visualize_side_by_side(rat_prob_map, info['ship'], path, info['bot'], title=f"Timesteps {timesteps}")

    return num_blocked_cell_detects, num_space_rat_pings, num_movements, timesteps

# Bot 2 that further optimizes to catch a moving rat by considering future movements of the rat and altering its planned path towards future movement.

def bot2_3(info, visualize, alpha):
    # Get initial bot position and ship configuration
    bot_r, bot_c = info['bot']
    rat_r, rat_c = info['rat']
    ship = info['empty_ship']
    d = len(info['ship'])

    # store all open cell coordinates in a list
    open_cells = []
    for r in range(d):
        for c in range(d):
            if ship[r][c] == 0:
                open_cells.append((r,c))

    # Initialize current positions
    curr_r, curr_c = bot_r, bot_c
    curr_rat_r, curr_rat_c = rat_r, rat_c

    # Generate helper variables
    neighbor_map, blocked_neighbors = create_neighbor_map(ship)
    set_possible_cells = {(r, c) for r in range(d) for c in range(d) if ship[r][c] == 0}

    # Initialize result variables
    num_movements = 0
    num_blocked_cell_detects = 0
    num_space_rat_pings = 0
    timesteps = 0

    if visualize: print("phase 1")

    # ----------------- PHASE 1: Localization -----------------
    while len(set_possible_cells) > 1:
        # Sense blocked neighbors
        num_curr_blocked_ns = neighbor_map[curr_r][curr_c]
        possible_cells = {cell for cell in set_possible_cells if neighbor_map[cell[0]][cell[1]] == num_curr_blocked_ns}
        num_blocked_cell_detects += 1
        timesteps += 1

        # Rat moves
        rat_move_choices = [(dr, dc) for dr, dc in directions if ship[curr_rat_r + dr][curr_rat_c + dc] == 0]
        info['ship'][curr_rat_r][curr_rat_c] = 0
        actual_rat_move = random.choice(rat_move_choices)
        curr_rat_r += actual_rat_move[0]
        curr_rat_c += actual_rat_move[1]
        info['ship'][curr_rat_r][curr_rat_c] = 3
        info['rat'] = (curr_rat_r, curr_rat_c)

        # Determine possible moves
        direction_c = {(0, 1): set(), (0, -1): set(), (-1, 0): set(), (1, 0): set()}
        for pcr, pcc in possible_cells:
            for dr, dc in directions:
                nr, nc = pcr + dr, pcc + dc
                if ship[nr][nc] != 0:
                    direction_c[(dr, dc)].add((pcr, pcc))

        # Randomly choose a direction and move
        best_dir = random.choice(directions)
        nr, nc = curr_r + best_dir[0], curr_c + best_dir[1]

        if ship[nr][nc] == 0:  # Successful move
            set_possible_cells = set(possible_cells).difference(direction_c[best_dir])
            curr_r, curr_c = nr, nc
            new_set_possible_cells = {(r + best_dir[0], c + best_dir[1]) for r, c in set_possible_cells}
            set_possible_cells = new_set_possible_cells
            info['bot'] = (curr_r, curr_c)
            info['ship'][bot_r][bot_c] = 0
            info['ship'][curr_r][curr_c] = 2
            bot_r, bot_c = curr_r, curr_c
        else:  # Unsuccessful move
            set_possible_cells = direction_c[best_dir]

        num_movements += 1
        timesteps += 1

        # Rat moves again
        rat_move_choices = [(dr, dc) for dr, dc in directions if ship[curr_rat_r + dr][curr_rat_c + dc] == 0]
        info['ship'][curr_rat_r][curr_rat_c] = 0
        actual_rat_move = random.choice(rat_move_choices)
        curr_rat_r += actual_rat_move[0]
        curr_rat_c += actual_rat_move[1]
        info['ship'][curr_rat_r][curr_rat_c] = 3
        info['rat'] = (curr_rat_r, curr_rat_c)

    if visualize: print("phase 2")

    # ----------------- PHASE 2: Rat Finding -----------------
    curr_r, curr_c = set_possible_cells.pop()
    info['ship'][bot_r][bot_c] = 0
    info['ship'][curr_r][curr_c] = 2
    info['bot'] = (curr_r, curr_c)
    bot_r, bot_c = curr_r, curr_c

    # Initialize probability map and transition matrix
    rat_prob_map = [[-1 if ship[r][c] == 1 else 0 for c in range(d)] for r in range(d)]
    T = compute_transition_matrix(ship, open_cells)
    num_open_cells = len(open_cells)
    uniform_prob_i = 1 / num_open_cells
    for r, c in open_cells:
        rat_prob_map[r][c] = uniform_prob_i

    # Initialize future probability map
    # key = cell, value = anticipated probability one time step in the future based on current probability map
    future_prob = {}
    for r, c in open_cells:
        total_prob = 0
        for nr, nc in T:  # Loop over all possible source cells in the transition matrix
            if (r, c) in T[(nr, nc)]:  # Check if (r, c) is a possible destination from (nr, nc)
                # anticipate probability of this cell in the future
                prob_from_nr_nc = rat_prob_map[nr][nc] * T[(nr, nc)][(r, c)]
                total_prob += prob_from_nr_nc
        future_prob[(r, c)] = total_prob

    path = []
    found = False

    # Initial ping
    ping_result = ping(info, alpha)
    num_space_rat_pings += 1
    timesteps += 1
    
    rat_move_choices = [(dr, dc) for dr, dc in directions if ship[curr_rat_r + dr][curr_rat_c + dc] == 0]
    info['ship'][curr_rat_r][curr_rat_c] = 0
    actual_rat_move = random.choice(rat_move_choices)
    curr_rat_r += actual_rat_move[0]
    curr_rat_c += actual_rat_move[1]
    info['ship'][curr_rat_r][curr_rat_c] = 3
    info['rat'] = (curr_rat_r, curr_rat_c)

    if ping_result == 'Found':
        return num_blocked_cell_detects, num_space_rat_pings, num_movements, timesteps

    # Update probabilities based on initial ping
    rat_prob_map[bot_r][bot_c] = 0
    summed_prob = 0
    if ping_result == 'True':
        for r, c in open_cells:
            prob_rat = math.e ** (-alpha * (abs(r - bot_r) + abs(c - bot_c) - 1))
            rat_prob_map[r][c] *= prob_rat
            summed_prob += rat_prob_map[r][c]
    else:
        for r, c in open_cells:
            rat_prob_map[r][c] *= (1 - math.e ** (-alpha * (abs(r - bot_r) + abs(c - bot_c) - 1)))
            summed_prob += rat_prob_map[r][c]

    for r, c in open_cells:
        rat_prob_map[r][c] /= summed_prob

    # update rat_prob_map based on rat random movement
    new_rat_prob_map = [[-1 if ship[r][c] == 1 else 0 for c in range(d)] for r in range(d)]
    
    for r, c in open_cells:
        for nr, nc in T[(r, c)]:
            new_rat_prob_map[nr][nc] += rat_prob_map[r][c] * T[(r, c)][(nr, nc)]
    
    rat_prob_map = new_rat_prob_map

    # Update anticipated rat probabilities one step in the future
    for r, c in open_cells:
        total_prob = 0
        for nr, nc in T:  # Loop over all possible source cells in the transition matrix
            if (r, c) in T[(nr, nc)]:  # Check if (r, c) is a possible destination from (nr, nc)
                # anticipate probability of this cell in the future
                prob_from_nr_nc = rat_prob_map[nr][nc] * T[(nr, nc)][(r, c)]
                total_prob += prob_from_nr_nc
        future_prob[(r, c)] = total_prob

    # Main rat-finding loop
    while not found:
        # Find cell with highest probability
        highest_rat_prob = 0
        highest_rat_prob_cell = (-1, -1)
        for r, c in open_cells:
            if rat_prob_map[r][c] > highest_rat_prob:
                highest_rat_prob = rat_prob_map[r][c]
                highest_rat_prob_cell = (r, c)
        
        # Plan shortest path using A* with predictive heuristic
        
        def astar_with_new_heuristic(start, rat_prob_map, map, target, neighbor_map, future_prob):
            def heuristic(cell):
                manhattan_dist = abs(cell[0] - target[0]) + abs(cell[1] - target[1])
                future_prob_weight = 100  # Weight for future probability
                future_prob_val = future_prob.get(cell, 0)
                # Ensure heuristic remains non-negative
                return max(0, manhattan_dist - future_prob_weight * future_prob_val)

            d = len(map)
            fringe = [(heuristic(start), start)]
            total_costs = {start: 0}
            prev = {start: None}
            visited = set()  # To prevent revisiting nodes unnecessarily

            while fringe:
                if visualize: print("stuck in fringe, ")
                _, curr = heapq.heappop(fringe)
                if curr == target:
                    if visualize: print("found target")
                    path = deque()
                    curr_p = curr
                    seen = set()  # Detect cycles during path reconstruction
                    max_steps = d * d  # Safeguard: limit to grid size
                    step = 0
                    while curr_p is not None and step < max_steps:
                        if curr_p in seen:
                            break
                        path.appendleft(curr_p)
                        seen.add(curr_p)
                        curr_p = prev[curr_p]
                        step += 1

                    return list(path)

                if curr in visited:
                    continue
                visited.add(curr)

                r, c = curr
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < d and 0 <= nc < d and map[nr][nc] != 1:
                        cost = total_costs[curr] + 1  # Base movement cost
                        # Adjust cost with probability, but ensure it stays non-negative
                        prob_adjust = min(cost, 50 * rat_prob_map[nr][nc] + 10 * future_prob.get((nr, nc), 0))
                        new_cost = cost - prob_adjust  # Cost >= 0 due to min()
                        if (nr, nc) not in total_costs or new_cost < total_costs[(nr, nc)]:
                            total_costs[(nr, nc)] = new_cost
                            est_total_cost = new_cost + heuristic((nr, nc))
                            heapq.heappush(fringe, (est_total_cost, (nr, nc)))
                            prev[(nr, nc)] = curr

            return [start]  # Return current position if no path exists
        

        path = astar_with_new_heuristic(info['bot'], rat_prob_map, info['empty_ship'], highest_rat_prob_cell, neighbor_map, future_prob)

        # Move along the path one step at a time
        for new_r, new_c in path[:10]:
            # Move bot
            info['ship'][bot_r][bot_c] = 0
            info['bot'] = (new_r, new_c)
            info['ship'][new_r][new_c] = 2
            bot_r, bot_c = new_r, new_c
            num_movements += 1
            timesteps += 1


            # Rat moves
            rat_move_choices = [(dr, dc) for dr, dc in directions if ship[curr_rat_r + dr][curr_rat_c + dc] == 0]
            info['ship'][curr_rat_r][curr_rat_c] = 0
            actual_rat_move = random.choice(rat_move_choices)
            curr_rat_r += actual_rat_move[0]
            curr_rat_c += actual_rat_move[1]
            info['ship'][curr_rat_r][curr_rat_c] = 3
            info['rat'] = (curr_rat_r, curr_rat_c)

            # Ping
            ping_result = ping(info, alpha)
            num_space_rat_pings += 1
            timesteps += 1

            if ping_result == 'Found':
                print("in found")
                print(f"Rat found at timestep {timesteps}, Bot: {info['bot']}, Rat: {info['rat']}")
                if visualize:
                    visualize_side_by_side(rat_prob_map, info['ship'], path, info['bot'], title=f"Rat Found at Timestep {timesteps}")
                return num_blocked_cell_detects, num_space_rat_pings, num_movements, timesteps

            # Update probabilities based on ping
            rat_prob_map[bot_r][bot_c] = 0
            summed_prob = 0
            if ping_result == 'True':
                for r, c in open_cells:
                    prob_rat = math.e ** (-alpha * (abs(r - bot_r) + abs(c - bot_c) - 1))
                    rat_prob_map[r][c] *= prob_rat
                    summed_prob += rat_prob_map[r][c]
            else:
                for r, c in open_cells:
                    rat_prob_map[r][c] *= (1 - math.e ** (-alpha * (abs(r - bot_r) + abs(c - bot_c) - 1)))
                    summed_prob += rat_prob_map[r][c]

            for r, c in open_cells:
                rat_prob_map[r][c] /= summed_prob

            # Update probabilities based on rat movement
            new_rat_prob_map = [[-1 if ship[r][c] == 1 else 0 for c in range(d)] for r in range(d)]
            for r, c in open_cells:
                for nr, nc in T[(r, c)]:
                    new_rat_prob_map[nr][nc] += rat_prob_map[r][c] * T[(r, c)][(nr, nc)]
            rat_prob_map = new_rat_prob_map


            # Rat moves again
            rat_move_choices = [(dr, dc) for dr, dc in directions if ship[curr_rat_r + dr][curr_rat_c + dc] == 0]
            info['ship'][curr_rat_r][curr_rat_c] = 0
            actual_rat_move = random.choice(rat_move_choices)
            curr_rat_r += actual_rat_move[0]
            curr_rat_c += actual_rat_move[1]
            info['ship'][curr_rat_r][curr_rat_c] = 3
            info['rat'] = (curr_rat_r, curr_rat_c)
            
        # Update future probabilities
        new_future_prob = {}
        for r, c in open_cells:
            total_prob = 0
            for nr, nc in T:  # Loop over all possible source cells in the transition matrix
                if (r, c) in T[(nr, nc)]:  # Check if (r, c) is a possible destination from (nr, nc)
                    # anticipate probability of this cell in the future
                    prob_from_nr_nc = rat_prob_map[nr][nc] * T[(nr, nc)][(r, c)]
                    total_prob += prob_from_nr_nc
            new_future_prob[(r, c)] = total_prob
        future_prob = new_future_prob

    
    return num_blocked_cell_detects, num_space_rat_pings, num_movements, timesteps



# A* algorithm to find shortest path from any start position to any end position 
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

# Modified version of A* that promotes visiting cells with high rat probability even though they may be longer by reducing cost at a cell in a way proportional to its rat probability
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

                cost -= (1000 * rat_prob_map[nr][nc])

                est_total_cost = cost + heuristic(child)
                if child not in total_costs:
                    prev[child] = curr[1]
                    total_costs[child] = cost
                    heapq.heappush(fringe, (est_total_cost, child))
    return []        

# Main for testing
def main():
    random.seed(10)

    og_info = init_ship(30)

    info_1 = copy.deepcopy(og_info)
    info_2 = copy.deepcopy(og_info)
    print("BOT", info_1['bot'])
    print("RAT", info_1['rat'])

    visualize_ship(og_info['ship'], None)

    num_blocked_cell_detects, num_space_rat_pings, num_movements, timesteps = bot1_2(info_1, True, 0.04)

    print("BOT 1 PEFORMANCE:")
    print("num_blocked_cell_detects", num_blocked_cell_detects),
    print("num_space_rat_pings", num_space_rat_pings)
    print("num_movements", num_movements)
    print("time_steps", timesteps)
        

    num_blocked_cell_detects, num_space_rat_pings, num_movements, timesteps = bot2_3(info_2, True, 0.04)

    print("BOT 2 PEFORMANCE:")
    print("num_blocked_cell_detects", num_blocked_cell_detects),
    print("num_space_rat_pings", num_space_rat_pings)
    print("num_movements", num_movements)
    print("time_steps", timesteps)

if __name__ == "__main__":
    main()