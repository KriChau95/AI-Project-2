# ship.py represents all the information to simulate the space vessel and contains bot functions for each bot

# Importing libraries for randomness, data structures, and data visualization
import random
import numpy as np
import heapq
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors 
from collections import deque, defaultdict
import copy

global directions
directions = [(0,1), (0,-1), (1,0), (-1,0)] # array to store adjacent directions needed during various traversal
diagonal_directions = [(0,1), (0,-1), (1,0), (-1,0), (-1,1), (-1,-1), (1,1), (1,-1)]
random.seed(30)

# Initializes the maze with bot, button, fire, and open and closed cells based on an input dimension d - number of rows and columns
def init_ship(dimension):
        
    d = dimension - 2
    
    # 0 = open cell
    # 1 = closed cell
    # 2 = bot
    # 3 = rat

    # initialize ship to size dimension
    ship = [[1] * d for _ in range(d)] 

    # open up a random cell on interior
    to_open = random.sample(range(1,d-1), 2) # range is set from 1 to d-1 to ensure we select from interior
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

        chosen_coordinate = (random.choice(list(single_neighbor))) # choose cell randomly
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


    # Condense all the information created within this function into a hashmap and return the hashmap

    ship.insert(0,[1] * dimension)
    ship.append([1] * dimension)

    for i in range(1,dimension-1):
        row = ship[i]
        new_row = [1] + row + [1]
        ship[i] = new_row
    
    # create sets that  store coordinates of all the open cells and all the closed cells
    open_cells = set()
    for r in range(dimension):
        for c in range(dimension):
            if ship[r][c] == 0:
                open_cells.add((r,c))
    
    print("num open cells", len(open_cells))

    empty_ship = copy.deepcopy(ship)

    # randomly place bot in one of the remaining open cells
    bot_r,bot_c = (random.choice(list(open_cells)))
    open_cells.remove((bot_r,bot_c))

    # randomly place button in one of the remaining open cels
    rat_r,rat_c = (random.choice(list(open_cells)))
    open_cells.remove((rat_r,rat_c))

    # modifying the cells in the 2D array to store the appropriate special objects - bot, fire, burron
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

def visualize_neighbor_map(map, title = ''):
    # hashmap that maps item in 2D ship array representation to corresponding color for visualization
    color_map = {
        -1: 'black',
        0: 'white',
        1: 'indianred',
        2: 'darkorange',  
        3: 'olivedrab',  
        4: 'mediumturquoise',
        5: 'dodgerblue',
        6: 'rebeccapurple',
        7: 'magenta',
        8: 'yellow'
    }
    
    d = len(map)

    # set up a numpy array to represent the img
    img = np.zeros((d, d, 3))
    
    # loop through the ship 2D array and set the corresponding color based on the value in the array and the color_map
    for i in range(d):
        for j in range(d):
            img[i][j] = mcolors.to_rgb(color_map[map[i][j]])  
    

    # display the graph
    plt.imshow(img, interpolation='nearest')
    plt.xticks([])
    plt.yticks([])

    # if a title is requested, set it
    if title != "":
        plt.title(title)
    
    # show the visualization
    plt.show()   


    

def visualize_ship(ship, path, title = ""): 

    # hashmap that maps item in 2D ship array representation to corresponding color for visualization
    color_map = {
        0: 'white', # Empty space
        1: 'black',  # Wall
        2: 'deepskyblue',   # Bot
        3: 'goldenrod'  # Rat
    }
    
    d = len(ship)

    # set up a numpy array to represent the img
    img = np.zeros((d, d, 3))
    
    # loop through the ship 2D array and set the corresponding color based on the value in the array and the color_map
    for i in range(d):
        for j in range(d):
            img[i][j] = mcolors.to_rgb(color_map[ship[i][j]])  
    
    # display the path by coloring in all cells from start of path to end of path orange
    if path is not None:
        for i in range(len(path)):
            r, c = path[i]
            img[r][c] = mcolors.to_rgb('orange')

    # display the graph
    plt.imshow(img, interpolation='nearest')
    plt.xticks([])
    plt.yticks([])

    # if a title is requested, set it
    if title != "":
        plt.title(title)
    
    # show the visualization
    plt.show()   

def visualize_possible_cells(ship, cells, title = ""): 

    # hashmap that maps item in 2D ship array representation to corresponding color for visualization
    color_map = {
        0: 'white', # Empty space
        1: 'black',  # Wall
        2: 'deepskyblue',   # Bot
    }
    
    d = len(ship)

    # set up a numpy array to represent the img
    img = np.zeros((d, d, 3))
    
    # loop through the ship 2D array and set the corresponding color based on the value in the array and the color_map
    for i in range(d):
        for j in range(d):
            img[i][j] = mcolors.to_rgb(color_map[ship[i][j]]) 

    for cell in cells:
        img[cell[0]][cell[1]] = mcolors.to_rgb('c') 
    

    # display the graph
    plt.imshow(img, interpolation='nearest')
    plt.xticks([])
    plt.yticks([])

    # if a title is requested, set it
    if title != "":
        plt.title(title)
    
    # show the visualization
    plt.show()   

def bot1(info, visualize):
    bot_r, bot_c = info['bot']
    ship = info['empty_ship']
    curr_r, curr_c = bot_r, bot_c
    neighbor_map, blocked_neighbors = create_neighbor_map(ship)
    num_curr_blocked_ns = neighbor_map[curr_r][curr_c]
    possible_cells = blocked_neighbors[num_curr_blocked_ns]
    prev_dirc = (1,1)

    while len(possible_cells) > 1:

        num_curr_blocked_ns = neighbor_map[curr_r][curr_c]
        
        print('curr_blocked_ns', num_curr_blocked_ns)

        visualize_possible_cells(ship, possible_cells)



        direction_o = {
            (0,1) : 0,
            (0,-1) : 0,
            (-1,0) : 0,
            (1,0) : 0
        }

        direction_c = {
            (0,1) : set(),
            (0,-1) : set(),
            (-1,0) : set(),
            (1,0) : set()
        }



        for pcr, pcc in possible_cells:
            # print(pcr, pcc)
            for dr, dc in directions:
                nr = pcr + dr
                nc = pcc + dc
                # print(nr,nc, ship[nr][nc])
                if ship[nr][nc] == 0:
                    direction_o[(dr,dc)] += 1
                else:
                    direction_c[(dr,dc)].add((pcr,pcc))
                    # print(f"adding pcr {pcr} and pcc {pcc} to directionc {(dr,dc)}")
        
        # print(direction_c)
        
        best_dir_arr = sorted(direction_o, key = lambda x: direction_o[x])
        print("sorted direction array", best_dir_arr, "previous direction", prev_dirc)
        if best_dir_arr[-1] == (-prev_dirc[0], -prev_dirc[1]):
            best_dir = best_dir_arr[-2]
        else:
            best_dir = best_dir_arr[-1]
            

        print(direction_o)
        nr,nc = curr_r + best_dir[0], curr_c + best_dir[1]

        if ship[nr][nc] == 0: # open
            set_possible_cells = set(possible_cells).difference(direction_c[best_dir])
            curr_r, curr_c = nr, nc
        else:
            set_possible_cells = direction_c[best_dir]
        
        visualize_possible_cells(ship, list(set_possible_cells))
        
        num_curr_blocked_ns = neighbor_map[curr_r][curr_c]

        possible_cells = set()

        for cr,cc in set_possible_cells:
            nr,nc = cr + best_dir[0], cc + best_dir[1]
            if num_curr_blocked_ns == neighbor_map[nr][nc]:
                possible_cells.add((nr,nc))



        prev_dirc = best_dir
         

        print(best_dir)
        print("YOOOOO",curr_r,curr_c)
        print("possible cells", possible_cells)



# Main for testing
def main():

    random.seed(15)
    info = init_ship(30)
    ship = info['ship']
    empty_ship = info['empty_ship']
    visualize_ship(ship, None)
    visualize_ship(empty_ship, None)
    neighbor_map, blocked_neighbors = create_neighbor_map(empty_ship)
    temp_sum = 0

    for key,value in blocked_neighbors.items():
        print(key, value, len(value))
        temp_sum += len(value)
        print()
    visualize_neighbor_map(neighbor_map)
    print("\n\n",temp_sum)
    bot1(info, visualize=False)

    num_closed = 0
    for r in range(30):
        for c in range(30):
            if ship[r][c] == 1:
                num_closed += 1
    print(num_closed)

# Run Main
if __name__ == "__main__":
    main()




# A* search algorithm implementation that takes in:
# start - tuple of (start row, start col) of where to start search
# map - contains information of map in current state - 2D array
# button - tuple of (button row, button col) of final destination
def astar(start, map, button):
    
    # heuristic used for A* - Manhattan distance between 2 points (x_1, y_1) and (x_2, y_2)
    # returns sum of absolute value in difference of x and absolute value in difference of y
    # takes in tuple cell1 (row, col) and returns Manhattan distance to goal - button
    def heuristic(cell1):
        return abs(cell1[0] - button[0]) + abs(cell1[1]-button[1])
    
    # initializing useful variables for A*
    d = len(map)
    fringe = []

    # more initialization of variables
    heapq.heappush(fringe, (heuristic(start),start))
    # items on the fringe (heap) will look like (23, (2,5))
    total_costs = dict()
    total_costs[start] = 0
    prev = dict()
    prev[start] = None

    # A* loop
    while fringe:
        
        # pop the cell with the lowest estimated cost
        curr = heapq.heappop(fringe)

        # curr = (heuristic((x,y)), (x,y))
        # curr pos = curr[1]
        # heuristic evaluated at curr pos = curr[0]

        # if we have reached the goal, reconstruct the path
        if curr[1] == button:
            
            curr_p = curr[1]
            path = deque()
            while curr_p != None:
                path.appendleft(curr_p)
                curr_p = prev[curr_p]
            return list(path)
        
        # get current cell's row and column
        r,c = curr[1]

        # explore neighboring cells
        for dr, dc in directions:
            nr, nc = r + dr, c + dc # calculating neighbor's coordinates
            child = (nr,nc) # child is tuple that represents neighbor's coordinates

            # check if neighbor is in bounds, not a wall, and not a fire
            if 0 <= nr < d and 0 <= nc < d and (map[nr][nc] != 1 and map[nr][nc] != -1):
                cost = total_costs[curr[1]] + 1

                # compute estimated total cost as sum of actual cost and heurisitc
                est_total_cost = cost + heuristic(child)

                # if this path to child is better (or if it's not visited yet), update
                if child not in total_costs:
                    prev[child] = curr[1]
                    total_costs[child] = cost
                    heapq.heappush(fringe, (est_total_cost, child))

    # if no path was found, return an empty list
    return []        