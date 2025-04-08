
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
