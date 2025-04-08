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


# New helper function to compute the transition matrix for rat movement
def compute_transition_matrix(ship):
    
    d = len(ship)
    
    open_cells = []
    for r in range(d):
        for c in range(d):
            if ship[r][c] == 0:
                open_cells.append((r,c))
    
    T = defaultdict(lambda: defaultdict(float))

    for r, c in open_cells:
        
        neighbors = []
        for dr, dc in directions:
            if 0 <= r + dr < d and 0 <= c + dc < d and ship [r + dr][c + dc] == 0:
                neighbors.append((r + dr, c + dc))

        prob = 1 / len(neighbors)
        
        for nr, nc in neighbors:
            T[(r, c)][(nr, nc)] = prob
    
    return T, open_cells

def bot1_2(info, visualize, alpha):
    
    # Get initial bot position and ship configuration
    bot_r, bot_c = info['bot']
    rat_r, rat_c = info['rat']
    ship = info['empty_ship']
    d = len(info['ship'])

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

    print("phase 1")

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

    print("phase 2")

    # ----------------- PHASE 2: Rat Finding -----------------
    
    curr_r, curr_c = set_possible_cells.pop()
    info['ship'][bot_r][bot_c] = 0
    info['ship'][curr_r][curr_c] = 2
    info['bot'] = (curr_r, curr_c)
    bot_r, bot_c = curr_r, curr_c

    # Initialize probability map and transition matrix
    rat_prob_map = [[-1 if ship[r][c] == 1 else 0 for c in range(d)] for r in range(d)]
    T, open_cells = compute_transition_matrix(ship)
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

def bot2_2(info, visualize, alpha):
    
    # Get initial bot position and ship configuration
    bot_r, bot_c = info['bot']
    rat_r, rat_c = info['rat']
    ship = info['empty_ship']
    d = len(info['ship'])

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

    print("phase 1")

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

    print("phase 2")

    # ----------------- PHASE 2: Rat Finding -----------------
    
    curr_r, curr_c = set_possible_cells.pop()
    info['ship'][bot_r][bot_c] = 0
    info['ship'][curr_r][curr_c] = 2
    info['bot'] = (curr_r, curr_c)
    bot_r, bot_c = curr_r, curr_c

    # Initialize probability map and transition matrix
    rat_prob_map = [[-1 if ship[r][c] == 1 else 0 for c in range(d)] for r in range(d)]
    T, open_cells = compute_transition_matrix(ship)
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
        path = astar_with_heuristic(info['bot'], rat_prob_map, info['empty_ship'], highest_rat_prob_cell)

        # Move along the path
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

                cost -= (100 * rat_prob_map[nr][nc])

                est_total_cost = cost + heuristic(child)
                if child not in total_costs:
                    prev[child] = curr[1]
                    total_costs[child] = cost
                    heapq.heappush(fringe, (est_total_cost, child))
    return []        


def main():
    random.seed(10)

    og_info = init_ship(30)

    info_1 = copy.deepcopy(og_info)

    print("BOT", info_1['bot'])
    print("RAT", info_1['rat'])

    visualize_ship(og_info['ship'], None)

    num_blocked_cell_detects, num_space_rat_pings, num_movements, timesteps = bot1_2(info_1, True, 0.1)

    print("BOT 1 PEFORMANCE:")
    print("num_blocked_cell_detects", num_blocked_cell_detects),
    print("num_space_rat_pings", num_space_rat_pings)
    print("num_movements", num_movements)
    print("time_steps", timesteps)
        

if __name__ == "__main__":
    main()






















