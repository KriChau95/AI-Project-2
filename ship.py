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

# Set backend to TkAgg at the start to avoid macosx issues
plt.switch_backend('TkAgg')

# setting up global variables that are used for adjacency in searches
global directions
directions = [(0,1), (0,-1), (1,0), (-1,0)] # array to store adjacent directions needed during various traversal
global diagonal_directions
diagonal_directions = [(0,1), (0,-1), (1,0), (-1,0), (-1,1), (-1,-1), (1,1), (1,-1)]

# the ping function takes in info - the current state of the ship, and alpha which dictates ping probability
# based on the current bot position and the current rat position, it simulates and returns a boolean represeting whether or not a ping occurred
def ping(info, alpha):

    bot_r, bot_c = info['bot']
    rat_r, rat_c = info['rat']
    
    # print('bot_r, bot_c:', bot_r, bot_c)
    # print('rat_r, rat_c:', rat_r, rat_c)

    def heuristic():
        return abs(bot_r - rat_r) + abs(bot_c - rat_c)
    
    prob_ping = math.e ** (-alpha * (heuristic() - 1))
    
    if random.uniform(0,1) < prob_ping:
        return True, prob_ping
    else:
        return False, prob_ping

# Bot 1 Implementation here
def bot1(info, visualize, alpha):
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

    # --- Frame storage for visualization ---
    if visualize:
        frames = []  # List to store frames
        # Initial state at timestep 0
        frames.append({
            'ship': copy.deepcopy(info['ship']),
            'rat_prob_map': copy.deepcopy(ship),  # Initially, no rat prob map, use empty ship
            'path': None,
            'bot_pos': info['bot'],
            'title': f"bot1 for alpha={alpha}, Timestep {timesteps}"
        })

    ## Phase 1: Localization
    while len(possible_cells) > 1:
        # Sense blocked neighbors (alternating action 1)
        num_curr_blocked_ns = neighbor_map[curr_r][curr_c]
        possible_cells = set()
        for cellr, cellc in set_possible_cells:
            if neighbor_map[cellr][cellc] == num_curr_blocked_ns:
                possible_cells.add((cellr, cellc))
        num_blocked_cell_detects += 1
        timesteps += 1

        # Capture frame after sensing
        if visualize:
            frames.append({
                'ship': copy.deepcopy(info['ship']),
                'rat_prob_map': copy.deepcopy(ship),  # Still no rat prob map in Phase 1
                'path': None,
                'bot_pos': info['bot'],
                'title': f"bot1 for alpha={alpha}, Timestep {timesteps} (Sensing Blocked Neighbors)"
            })

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
        if best_dir_arr[-1] == (-prev_dirc[0], -prev_dirc[1]):
            best_dir = best_dir_arr[-2]
        else:
            best_dir = best_dir_arr[-1]
    
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
            num_movements += 1
            info['bot'] = (curr_r, curr_c)  # Update bot position in info
            info['ship'][bot_r][bot_c] = 0  # Clear old position
            info['ship'][curr_r][curr_c] = 2  # Set new position
            bot_r, bot_c = curr_r, curr_c  # Update bot_r, bot_c
        else:  # Blocked, no movement
            set_possible_cells = direction_c[best_dir]
        
        prev_dirc = best_dir
        timesteps += 1

        # Capture frame after move attempt
        if visualize:
            frames.append({
                'ship': copy.deepcopy(info['ship']),
                'rat_prob_map': copy.deepcopy(ship),  # Still no rat prob map in Phase 1
                'path': None,
                'bot_pos': info['bot'],
                'title': f"bot1 for alpha={alpha}, Timestep {timesteps} (Move)"
            })

    ## Phase 2: Rat finding
    curr_r, curr_c = set_possible_cells.pop()
    info['ship'][bot_r][bot_c] = 0
    info['ship'][curr_r][curr_c] = 2
    info['bot'] = (curr_r, curr_c)
    bot_r, bot_c = curr_r, curr_c  # Sync bot_r, bot_c with new position

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

    # Capture initial rat prob map state
    if visualize:
        frames.append({
            'ship': copy.deepcopy(info['ship']),
            'rat_prob_map': copy.deepcopy(rat_prob_map),
            'path': None,
            'bot_pos': info['bot'],
            'title': f"bot1 for alpha={alpha}, Timestep {timesteps} (Phase 2 Start)"
        })

    path = []

    def update_rat_prob(rat_prob_map):
        nonlocal num_space_rat_pings, timesteps
        ping_result, prob_ping = ping(info, alpha)
        num_space_rat_pings += 1
        timesteps += 1

        summed_prob = 0
        if ping_result:
            for (r,c) in open_cells:
                prob_rat = math.e ** (-alpha * ((abs(r - bot_r) + abs(c - bot_c))-1))
                rat_prob_map[r][c] *= prob_rat
                summed_prob += rat_prob_map[r][c]
            for (r,c) in open_cells:
                rat_prob_map[r][c] /= summed_prob
        else:
            for (r,c) in open_cells:
                rat_prob_map[r][c] *= (1 - math.e ** (-alpha * ((abs(r - bot_r) + abs(c - bot_c))-1)))
                summed_prob += rat_prob_map[r][c]
            for (r,c) in open_cells:
                rat_prob_map[r][c] /= summed_prob

        # Capture frame after ping
        if visualize:
            frames.append({
                'ship': copy.deepcopy(info['ship']),
                'rat_prob_map': copy.deepcopy(rat_prob_map),
                'path': copy.deepcopy(path),
                'bot_pos': info['bot'],
                'title': f"bot1 for alpha={alpha}, Timestep {timesteps} (Ping)"
            })

    # Initial ping before the loop
    update_rat_prob(rat_prob_map)
    found = False

    while not found:
        highest_rat_prob = 0
        highest_rat_prob_cell = (-1,-1)
        for i in range(len(rat_prob_map)):
            for j in range(len(rat_prob_map)):
                if rat_prob_map[i][j] > highest_rat_prob:
                    highest_rat_prob = rat_prob_map[i][j]
                    highest_rat_prob_cell = (i,j)

        path = astar(info['bot'], info['empty_ship'], highest_rat_prob_cell)
        for new_r, new_c in path:
            # Move
            info['ship'][bot_r][bot_c] = 0  # Clear old position
            info['bot'] = (new_r, new_c)
            info['ship'][new_r][new_c] = 2  # Set new position
            bot_r, bot_c = new_r, new_c  # Update bot_r, bot_c
            num_movements += 1
            timesteps += 1

            # Capture frame after move
            if visualize:
                frames.append({
                    'ship': copy.deepcopy(info['ship']),
                    'rat_prob_map': copy.deepcopy(rat_prob_map),
                    'path': copy.deepcopy(path),
                    'bot_pos': info['bot'],
                    'title': f"bot1 for alpha={alpha}, Timestep {timesteps} (Move)"
                })

            if info['bot'] == info['rat']:
                found = True
            else:
                rat_prob_map[bot_r][bot_c] = 0

            if not found:  
                update_rat_prob(rat_prob_map)

    if visualize:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        def animate(frame_idx):
            ax1.clear()
            ax2.clear()
            frame = frames[frame_idx]
            img_ship = visualize_ship(frame['ship'], frame['path'], "Ship State", show=False)
            ax1.imshow(img_ship, interpolation='nearest')
            ax1.set_xticks([])
            ax1.set_yticks([])
            ax1.set_title("Ship State")
            img_prob = visualize_rat_prob_map(frame['rat_prob_map'], None, frame['title'], bot_pos=frame['bot_pos'], show=False)
            ax2.imshow(img_prob, interpolation='nearest')
            ax2.set_xticks([])
            ax2.set_yticks([])
            ax2.set_title(frame['title'])

        ani = animation.FuncAnimation(fig, animate, frames=len(frames), interval=500, repeat=False)
        ani.save(f'proj2-bot1-a={alpha}.mp4', writer='ffmpeg', fps=2)  
        # plt.show()
        # plt.ion()
    
    return num_blocked_cell_detects, num_space_rat_pings, num_movements, timesteps



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
    prev_dirc = (1, 1)
    
    # Initialize result variables
    num_movements = 0
    num_blocked_cell_detects = 0
    num_space_rat_pings = 0
    timesteps = 0  # Added to track total actions (sense, ping, move)

    # Frame storage for visualization
    if visualize:
        frames = []  # List to store frames
        # Initial state at timestep 0
        frames.append({
            'ship': copy.deepcopy(info['ship']),
            'rat_prob_map': copy.deepcopy(ship),  # Initially, no rat prob map, use empty ship
            'path': None,
            'bot_pos': info['bot'],
            'title': f"bot2 for alpha={alpha}, Timestep {timesteps}"
        })

    ## Phase 1: Localization (unchanged from bot1)
    while len(possible_cells) > 1:
        # Sense blocked neighbors
        num_curr_blocked_ns = neighbor_map[curr_r][curr_c]
        possible_cells = set()
        for cellr, cellc in set_possible_cells:
            if neighbor_map[cellr][cellc] == num_curr_blocked_ns:
                possible_cells.add((cellr, cellc))
        num_blocked_cell_detects += 1
        timesteps += 1

        # Capture frame after sensing
        if visualize:
            frames.append({
                'ship': copy.deepcopy(info['ship']),
                'rat_prob_map': copy.deepcopy(ship),  # No rat prob map in Phase 1
                'path': None,
                'bot_pos': info['bot'],
                'title': f"bot2 for alpha={alpha}, Timestep {timesteps}pard (Sense)"
            })

        # Determine move direction
        direction_o = {(0, 1): 0, (0, -1): 0, (-1, 0): 0, (1, 0): 0}
        direction_c = {(0, 1): set(), (0, -1): set(), (-1, 0): set(), (1, 0): set()}

        for pcr, pcc in possible_cells:
            for dr, dc in directions:
                nr = pcr + dr
                nc = pcc + dc
                if ship[nr][nc] == 0:
                    direction_o[(dr, dc)] += 1
                else:
                    direction_c[(dr, dc)].add((pcr, pcc))
        
        best_dir_arr = sorted(direction_o, key=lambda x: direction_o[x])
        if best_dir_arr[-1] == (-prev_dirc[0], -prev_dirc[1]):
            best_dir = best_dir_arr[-2]
        else:
            best_dir = best_dir_arr[-1]
    
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
            num_movements += 1
            info['ship'][bot_r][bot_c] = 0  # Clear old position
            info['ship'][curr_r][curr_c] = 2  # Set new position
            info['bot'] = (curr_r, curr_c)
            bot_r, bot_c = curr_r, curr_c  # Update bot_r, bot_c
        else:
            set_possible_cells = direction_c[best_dir]
        
        prev_dirc = best_dir
        timesteps += 1

        # Capture frame after move attempt
        if visualize:
            frames.append({
                'ship': copy.deepcopy(info['ship']),
                'rat_prob_map': copy.deepcopy(ship),  # No rat prob map in Phase 1
                'path': None,
                'bot_pos': info['bot'],
                'title': f"bot2 for alpha={alpha}, Timestep {timesteps} (Move)"
            })

    ## Phase 2: Rat finding
    curr_r, curr_c = set_possible_cells.pop()
    info['ship'][bot_r][bot_c] = 0
    info['ship'][curr_r][curr_c] = 2
    info['bot'] = (curr_r, curr_c)
    bot_r, bot_c = curr_r, curr_c  # Sync bot_r, bot_c

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
                open_cells.add((i, j))

    # Capture initial rat prob map state
    if visualize:
        frames.append({
            'ship': copy.deepcopy(info['ship']),
            'rat_prob_map': copy.deepcopy(rat_prob_map),
            'path': None,
            'bot_pos': info['bot'],
            'title': f"bot2 for alpha={alpha}, Timestep {timesteps} (Phase 2 Start)"
        })

    path = []

    def update_rat_prob(rat_prob_map):
        nonlocal num_space_rat_pings, timesteps
        ping_result, prob_ping = ping(info, alpha)
        num_space_rat_pings += 1
        timesteps += 1

        summed_prob = 0
        if ping_result:
            for (r, c) in open_cells:
                prob_rat = math.e ** (-alpha * ((abs(r - bot_r) + abs(c - bot_c)) - 1))
                rat_prob_map[r][c] *= prob_rat
                summed_prob += rat_prob_map[r][c]
            for (r, c) in open_cells:
                rat_prob_map[r][c] /= summed_prob
        else:
            for (r, c) in open_cells:
                rat_prob_map[r][c] *= (1 - math.e ** (-alpha * ((abs(r - bot_r) + abs(c - bot_c)) - 1)))
                summed_prob += rat_prob_map[r][c]
            for (r, c) in open_cells:
                rat_prob_map[r][c] /= summed_prob

        # Capture frame after ping
        if visualize:
            frames.append({
                'ship': copy.deepcopy(info['ship']),
                'rat_prob_map': copy.deepcopy(rat_prob_map),
                'path': copy.deepcopy(path),
                'bot_pos': info['bot'],
                'title': f"bot2 for alpha={alpha}, Timestep {timesteps} (Ping)"
            })

    # Initial ping before the loop
    update_rat_prob(rat_prob_map)
    found = False

    while not found:
        bot_r, bot_c = info['bot']  # Get current bot position
        highest_rat_prob = 0
        highest_rat_prob_cell = (-1, -1)

        for i in range(len(rat_prob_map)):
            for j in range(len(rat_prob_map)):
                if rat_prob_map[i][j] > 0:  
                    manhattan_dist = abs(i - bot_r) + abs(j - bot_c)
                    if manhattan_dist <= 10 and rat_prob_map[i][j] > highest_rat_prob:
                        highest_rat_prob = rat_prob_map[i][j]
                        highest_rat_prob_cell = (i, j)

        path = astar_with_heuristic(info['bot'], rat_prob_map, info['empty_ship'], highest_rat_prob_cell)
        index = 0
        for new_r, new_c in path:
            # Move
            info['ship'][bot_r][bot_c] = 0  # Clear old position
            info['bot'] = (new_r, new_c)
            info['ship'][new_r][new_c] = 2  # Set new position
            bot_r, bot_c = new_r, new_c
            num_movements += 1
            timesteps += 1

            # Capture frame after move
            if visualize:
                frames.append({
                    'ship': copy.deepcopy(info['ship']),
                    'rat_prob_map': copy.deepcopy(rat_prob_map),
                    'path': copy.deepcopy(path),
                    'bot_pos': info['bot'],
                    'title': f"bot2 for alpha={alpha}, Timestep {timesteps} (Move)"
                })

            if info['bot'] == info['rat']:
                found = True
            else:
                rat_prob_map[bot_r][bot_c] = 0

            # Regular ping after move
            if not found:
                update_rat_prob(rat_prob_map)

            # Extra pings every 3rd move
            if (index % 3) == 0 and not found:
                print(f"Extra pings at index {index}, len path {len(path)}, timestep {timesteps}")
                update_rat_prob(rat_prob_map)
                update_rat_prob(rat_prob_map)

            index += 1
            if found:
                break

    if visualize:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        def animate(frame_idx):
            ax1.clear()
            ax2.clear()
            frame = frames[frame_idx]
            img_ship = visualize_ship(frame['ship'], frame['path'], "Ship State", show=False)
            ax1.imshow(img_ship, interpolation='nearest')
            ax1.set_xticks([])
            ax1.set_yticks([])
            ax1.set_title("Ship State")
            img_prob = visualize_rat_prob_map(frame['rat_prob_map'], None, frame['title'], bot_pos=frame['bot_pos'], show=False)
            ax2.imshow(img_prob, interpolation='nearest')
            ax2.set_xticks([])
            ax2.set_yticks([])
            ax2.set_title(frame['title'])

        ani = animation.FuncAnimation(fig, animate, frames=len(frames), interval=500, repeat=False)
        ani.save(f'proj2-bot2-a={alpha}.mp4', writer='ffmpeg', fps=2)  
        plt.show()
        plt.ion()
    
    return num_blocked_cell_detects, num_space_rat_pings, num_movements, timesteps


def visualize_ship(ship, path, title="", show=True): 
    color_map = {
        0: 'white', # Empty space
        1: 'black',  # Wall
        2: 'deepskyblue',   # Bot
        3: 'goldenrod'  # Rat
    }
    
    d = len(ship)
    img = np.zeros((d, d, 3))
    
    for i in range(d):
        for j in range(d):
            img[i][j] = mcolors.to_rgb(color_map[ship[i][j]])  
    
    if path is not None:
        for i in range(len(path)):
            r, c = path[i]
            img[r][c] = mcolors.to_rgb('orange')

    if show:
        plt.imshow(img, interpolation='nearest')
        plt.xticks([])
        plt.yticks([])
        if title:
            plt.title(title)
        plt.show()
    return img  # Return image data for animation

def visualize_rat_prob_map(rat_prob_map, path=None, title="", bot_pos=None, show=True):
    color_map = {
        0: 'purple',   # Definitely not here
        -1: 'black',   # Wall
    }
    
    d = len(rat_prob_map)
    img = np.zeros((d, d, 3))
    
    prob_values = [rat_prob_map[i][j] for i in range(d) for j in range(d) if 0 < rat_prob_map[i][j] <= 1]
    
    if prob_values:
        min_val, max_val = min(prob_values), max(prob_values)
    else:
        min_val, max_val = 0, 1

    cmap = plt.cm.coolwarm

    for i in range(d):
        for j in range(d):
            if rat_prob_map[i][j] in color_map:
                img[i][j] = mcolors.to_rgb(color_map[rat_prob_map[i][j]])
            elif 0 < rat_prob_map[i][j] <= 1:
                val = rat_prob_map[i][j]
                normalized_val = (val - min_val) / (max_val - min_val) if max_val > min_val else val
                adjusted_val = np.sqrt(normalized_val)
                img[i, j] = cmap(adjusted_val)[:3]
    
    if path is not None:
        for r, c in path:
            img[r, c] = mcolors.to_rgb('orange')

    # Highlight bot's current position in green
    if bot_pos is not None:
        r, c = bot_pos
        img[r, c] = mcolors.to_rgb('green')

    if show:
        plt.imshow(img, interpolation='nearest')
        plt.xticks([])
        plt.yticks([])
        if title:
            plt.title(title)
        plt.show()
    return img  # Return image data for animation

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

    PURPLE_CELL_PENALTY = 15

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

                if rat_prob_map[nr][nc] == 0:
                    cost += PURPLE_CELL_PENALTY

                est_total_cost = cost + heuristic(child)
                if child not in total_costs:
                    prev[child] = curr[1]
                    total_costs[child] = cost
                    heapq.heappush(fringe, (est_total_cost, child))
    return []        


def main():
    random.seed(10)
    info = init_ship(30)
    ship = info['ship']
    empty_ship = info['empty_ship']
    neighbor_map, blocked_neighbors = create_neighbor_map(empty_ship)
    num_blocked_cell_detects, num_space_rat_pings, num_movements, timesteps = bot1(info, visualize=True, alpha=0.05)
    print("num_blocked_cell_detects", num_blocked_cell_detects),
    print("num_space_rat_pings", num_space_rat_pings)
    print("num_movements", num_movements)

    

if __name__ == "__main__":
    main()