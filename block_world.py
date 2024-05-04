import pygame
import random
import numpy as np
from collections import deque
import heapq

BLOCKTYPES = 5

# - returns the square root of the sum of the squares of the coordinate differences
# - distance between two points in the plane
def euclidean_distance(a, b):
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return (dx**2 + dy**2)**0.5

# - výpočet manhattanské vzdálenosti
# - vrací součet absolutních hodnot rozdílů souřadnic
# - vzdálenost mezi dvěma body v rovině
def manhattan_distance(a, b):
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return dx + dy

# - state space search algorithm
# - returns the path and the list of expanded nodes
def greedy_best_first_search(start, goal, env):
    # queue of pairs (priority, node)
    frontier = [(manhattan_distance(start, goal), start)]
    # dictionary for recording predecessors
    came_from = {start: None}
    # set for recording expanded nodes
    expanded_nodes = set()

    while frontier:
        # get the node with the lowest priority
        _, current = heapq.heappop(frontier)
        expanded_nodes.add(current)

        # if the goal is reached, break the loop
        if current == goal:
            break
    
        # add neighbors to the queue
        # neighbors are added only if they have not been expanded yet
        for next in env.get_neighbors(*current):
            # if the neighbor has not been visited yet
            # calculate the priority and add the neighbor to the queue
            if next not in expanded_nodes:
                priority = manhattan_distance(goal, next)
                heapq.heappush(frontier, (priority, next))
                came_from[next] = current
    
    # reconstruct the path
    # start with the goal and follow the predecessors until the start is reached
    path = []
    while current is not None:
        path.append(current)
        current = came_from[current]
    path.reverse()

    return path, list(expanded_nodes)

# - state space search algorithm
# - returns the path and the list of expanded nodes
# - finds the shortest path in a weighted graph
def dijkstra(start, goal, env):
    frontier = [(0, start)]
    came_from = {start: None}
    cost_so_far = {start: 0}
    expanded_nodes = set()
    
    # get the node with the lowest cost
    # if the goal is reached, break the loop
    while frontier:
        _, current = heapq.heappop(frontier)
        expanded_nodes.add(current)

        if current == goal:
            break
    
        # add neighbors to the queue
        # neighbors are added only if they have not been expanded yet
        # if the cost of the path to the neighbor is lower than the current cost
        for next in env.get_neighbors(*current):
            new_cost = cost_so_far[current] + 1
            # if the neighbor has not been visited yet or the new cost is lower
            # update the cost and add the neighbor to the queue
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost
                heapq.heappush(frontier, (priority, next))
                came_from[next] = current

    path = []
    while current is not None:
        path.append(current)
        current = came_from[current]
    path.reverse()

    return path, list(expanded_nodes)

# - state space search algorithm
# - returns the path and the list of expanded nodes
# - finds the shortest path in a weighted graph
def a_star(start, goal, env):
    frontier = [(0, start)]
    came_from = {start: None}
    cost_so_far = {start: 0}
    heuristic_cache = {start: manhattan_distance(start, goal)}
    expanded_nodes = set()
    
    # get the node with the lowest cost
    while frontier:
        _, current = heapq.heappop(frontier)
        expanded_nodes.add(current)

        if current == goal:
            break
    
        # add neighbors to the queue
        # neighbors are added only if they have not been expanded yet
        # if the cost of the path to the neighbor is lower than the current cost
        # the cost is calculated as the sum of the cost so far and the heuristic
        for next in env.get_neighbors(*current):
            new_cost = cost_so_far[current] + 1
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                # if the heuristic value for the neighbor has not been calculated yet, calculate it
                # the heuristic value is the manhattan distance between the neighbor and the goal
                if next not in heuristic_cache:
                    heuristic_cache[next] = manhattan_distance(goal, next)
                priority = new_cost + heuristic_cache[next]
                heapq.heappush(frontier, (priority, next))
                came_from[next] = current

    path = []
    while current is not None:
        path.append(current)
        current = came_from[current]
    path.reverse()

    return path, list(expanded_nodes)


# třída reprezentující prostředí
class Env:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.arr = np.zeros((height, width), dtype=int)
        self.startx = 0
        self.starty = 0
        self.goalx = width-1
        self.goaly = height-1
        
    def is_valid_xy(self, x, y):      
        if x >= 0 and x < self.width and y >= 0 and y < self.height and self.arr[y, x] == 0:
            return True
        return False 
        
    def set_start(self, x, y):
        if self.is_valid_xy(x, y):
            self.startx = x
            self.starty = y
            
    def set_goal(self, x, y):
        if self.is_valid_xy(x, y):
            self.goalx = x
            self.goaly = y

        
    def is_empty(self, x, y):
        if self.arr[y, x] == 0:
            return True
        return False
    
    def add_block(self, x, y):
        if self.arr[y, x] == 0:
            r = random.randint(1, BLOCKTYPES)
            self.arr[y, x] = r
                
    def get_neighbors(self, x, y):
        l = []
        if x-1 >= 0 and self.arr[y, x-1] == 0:
            l.append((x-1, y))
        if x+1 < self.width and self.arr[y, x+1] == 0:
            l.append((x+1, y))
        if y-1 >= 0 and self.arr[y-1, x] == 0:
            l.append((x, y-1))
        if y+1 < self.height and self.arr[y+1, x] == 0:
            l.append((x, y+1))
        return l
        
    def get_tile_type(self, x, y):
        return self.arr[y, x]
    
    # vrací dvojici 1. frontu dvojic ze startu do cíle, 2. seznam dlaždic
    # k zobrazení - hodí se např. pro zvýraznění cesty, nebo expandovaných uzlů
    # start a cíl se nastaví pomocí set_start a set_goal
    # <------    ZDE vlastní metoda
    
    def path_planner(self, start, goal, algorithm):
        if start == goal:
            return [], []
        
        if algorithm == 'greedy':
            return greedy_best_first_search(start, goal, self)
        elif algorithm == 'dijkstra':
            return dijkstra(start, goal, self)
        elif algorithm == 'a_star':
            return a_star(start, goal, self)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    
# třída reprezentující ufo        
class Ufo:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.path = deque()
        self.tiles = []
    
    # přemístí ufo na danou pozici - nejprve je dobré zkontrolovat u prostředí, 
    # zda je pozice validní
    def move(self, x, y):
        self.x = x
        self.y = y
    
    # reaktivní navigace <------------------------ !!!!!!!!!!!! ZDE DOPLNIT
    def reactive_go(self, env):
        r = random.random()
        dx = 0
        dy = 0
        if r > 0.5: 
            r = random.random()
            if r < 0.5:
                dx = -1
            else:
                dx = 1
        else:
            r = random.random()
            if r < 0.5:
                dy = -1
            else:
                dy = 1
        return (self.x + dx, self.y + dy)
        
    # nastaví cestu k vykonání 
    def set_path(self, p, t=[]):
        # p je fronta dvojic
        # t je seznam dlaždic k zobrazení
        self.path = deque(p)
        self.tiles = t
    
    # vykoná naplánovanou cestu, v každém okamžiku na vyzvání vydá další
    # way point 
    # pokud je cesta prázdná, vrací (-1, -1)
    def execute_path(self):
        if self.path:
            return self.path.popleft()
        return (-1, -1)

# definice prostředí -----------------------------------
TILESIZE = 50
#<------    definice prostředí a překážek !!!!!!
WIDTH = 12
HEIGHT = 9
env = Env(WIDTH, HEIGHT)
env.add_block(1, 1)
env.add_block(2, 2)
env.add_block(3, 3)
env.add_block(4, 4)
env.add_block(5, 5)
env.add_block(6, 6)
env.add_block(7, 7)
env.add_block(8, 8)
env.add_block(0, 8)
env.add_block(11, 1)
env.add_block(11, 6)
env.add_block(1, 3)
env.add_block(2, 4)
env.add_block(4, 5)
env.add_block(2, 6)
env.add_block(3, 7)
env.add_block(4, 8)
env.add_block(0, 8)
env.add_block(1, 8)
env.add_block(2, 8)
env.add_block(3, 5)
env.add_block(4, 8)
env.add_block(5, 6)
env.add_block(6, 4)
env.add_block(7, 2)
env.add_block(8, 1)

# pozice ufo <--------------------------
ufo = Ufo(env.startx, env.starty)
WIN = pygame.display.set_mode((env.width * TILESIZE, env.height * TILESIZE))
pygame.display.set_caption("Block world")
pygame.font.init()
WHITE = (255, 255, 255)
FPS = 2

# pond, tree, house, car
BOOM_FONT = pygame.font.SysFont("comicsans", 100)   
LEVEL_FONT = pygame.font.SysFont("comicsans", 20)   
TILE_IMAGE = pygame.image.load("tile.jpg")
MTILE_IMAGE = pygame.image.load("markedtile.jpg")
HOUSE1_IMAGE = pygame.image.load("house1.jpg")
HOUSE2_IMAGE = pygame.image.load("house2.jpg")
HOUSE3_IMAGE = pygame.image.load("house3.jpg")
TREE1_IMAGE  = pygame.image.load("tree1.jpg")
TREE2_IMAGE  = pygame.image.load("tree2.jpg")
UFO_IMAGE = pygame.image.load("ufo.jpg")
FLAG_IMAGE = pygame.image.load("flag.jpg")
TILE = pygame.transform.scale(TILE_IMAGE, (TILESIZE, TILESIZE))
MTILE = pygame.transform.scale(MTILE_IMAGE, (TILESIZE, TILESIZE))
HOUSE1 = pygame.transform.scale(HOUSE1_IMAGE, (TILESIZE, TILESIZE))
HOUSE2 = pygame.transform.scale(HOUSE2_IMAGE, (TILESIZE, TILESIZE))
HOUSE3 = pygame.transform.scale(HOUSE3_IMAGE, (TILESIZE, TILESIZE))
TREE1 = pygame.transform.scale(TREE1_IMAGE, (TILESIZE, TILESIZE))
TREE2 = pygame.transform.scale(TREE2_IMAGE, (TILESIZE, TILESIZE))
UFO = pygame.transform.scale(UFO_IMAGE, (TILESIZE, TILESIZE))
FLAG = pygame.transform.scale(FLAG_IMAGE, (TILESIZE, TILESIZE))

def draw_window(ufo, env):
    for i in range(env.width):
        for j in range(env.height):
            t = env.get_tile_type(i, j)
            if t == 1:
                WIN.blit(TREE1, (i*TILESIZE, j*TILESIZE))
            elif t == 2:
                WIN.blit(HOUSE1, (i*TILESIZE, j*TILESIZE))
            elif t == 3:
                WIN.blit(HOUSE2, (i*TILESIZE, j*TILESIZE))
            elif t == 4:
                WIN.blit(HOUSE3, (i*TILESIZE, j*TILESIZE))  
            elif t == 5:
                WIN.blit(TREE2, (i*TILESIZE, j*TILESIZE))     
            else:
                WIN.blit(TILE, (i*TILESIZE, j*TILESIZE))
    for (x, y) in ufo.tiles:
        WIN.blit(MTILE, (x*TILESIZE, y*TILESIZE))
    WIN.blit(FLAG, (env.goalx * TILESIZE, env.goaly * TILESIZE))        
    WIN.blit(UFO, (ufo.x * TILESIZE, ufo.y * TILESIZE))
    pygame.display.update()
    
def main():
    #  <------------   nastavení startu a cíle prohledávání !!!!!!!!!!
    env.set_start(0, 0)
    env.set_goal(9, 7)
    # p, t = env.path_planner((0, 0), (9, 7), 'a_star')
    # p, t = env.path_planner((0, 0), (9, 7), 'dijkstra')
    p, t = env.path_planner((0, 0), (9, 7), 'greedy')
    ufo.set_path(p, t)
    # ---------------------------------------------------
    clock = pygame.time.Clock()
    run = True
    go = False

    steps = 0  # Initialize a counter for steps
    
    while run:  
        clock.tick(FPS)
        # <---- reaktivní pohyb dokud nedojde do cíle 
        # nebo dokud není cesta prázdná
        # v každém kroku se zobrazí okno
        # a počká se na událost
        if (ufo.x != env.goalx) or (ufo.y != env.goaly):        
            # x, y = ufo.reactive_go(env)
            x, y = ufo.execute_path()
            # Check if the UFO is at a valid position
            # If it is, move the UFO and increment the step counter
            if env.is_valid_xy(x, y):
                # Move the UFO to the new position
                ufo.move(x, y)
                steps += 1
                print(f"Step {steps}: UFO is at ({ufo.x}, {ufo.y})")  # Print the step number and position
        elif(ufo.x == env.goalx) and (ufo.y == env.goaly):
                print("UFO has reached the flag. Stopping the program.")
                # break
        else:
                print('[', x, ',', y, ']', "wrong coordinate !")
                        
        draw_window(ufo, env)
        
        # If the user closes the window, stop the program
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
    pygame.quit()    

if __name__ == "__main__":
    main()