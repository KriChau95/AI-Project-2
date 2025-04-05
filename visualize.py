
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors 
import matplotlib.animation as animation


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