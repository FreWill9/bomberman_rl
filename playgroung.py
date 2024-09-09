import numpy as np
import torch
from collections import namedtuple, deque
from random import shuffle

experiment_state = {'round': 1,

                    'step': 9,

                    'field': np.array(
                        [[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                         [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                         [-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1],
                         [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                         [-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1],
                         [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                         [-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1],
                         [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                         [-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1],
                         [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                         [-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1],
                         [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                         [-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1],
                         [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                         [-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1],
                         [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                         [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]),

                    'self': ('random_agent', 23, True, (np.int64(1), np.int64(1))),

                    'others': [('peaceful_agent', 3, True, (np.int64(2), np.int64(15))),
                               ('coin_collector_agent', 9, False, (np.int64(15), np.int64(15)))],

                    'bombs': [((np.int64(15), np.int64(15)), 2)],

                    'coins': [(np.int64(1), np.int64(1))],

                    'user_input': None,

                    'explosion_map': np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                               [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])}

Memory = namedtuple('Memory',
                    ('state', 'action', 'reward', 'next_state', 'done'))

coordinate_history = deque([(3, 3), (2, 2), (3, 3), (4, 2), (3, 3)], 20)


def build_bomb_map(game_state: dict):
    bomb_map = np.ones(game_state['field'].shape) * 5
    for (xb, yb), t in game_state['bombs']:
        for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
            if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                bomb_map[i, j] = min(bomb_map[i, j], t)

    return bomb_map


def tile_value(game_state: dict, coord: (int, int), coordinate_history: deque) -> float:
    bomb_map = build_bomb_map(game_state)
    explosion_map = game_state['explosion_map']
    bomb_coord = [xy for (xy, t) in game_state['bombs']]
    opp_coord = [xy for (n, s, b, xy) in game_state['others']]
    temp = 0.0

    match game_state['field'][coord[0], coord[1]]:
        case 0:
            temp = 0.0
        case -1:
            temp = -0.45
        case 1:
            temp = 0.5
    if coordinate_history.count((coord[0], coord[1])) > 2:
        temp = -0.4
    if (coord[0], coord[1]) in opp_coord:
        temp = -0.5
    if (coord[0], coord[1]) in game_state['coins']:
        temp = 1.0
    if bomb_map[coord[0], coord[1]] < 5:
        temp = -0.9
    if explosion_map[coord[0], coord[1]] > 0:
        temp = -1.0
    if (coord[0], coord[1]) in bomb_coord:
        temp = -1.0

    return temp


def look_for_targets(free_space, start, targets, logger=None):
    """
    Find direction of the closest target that can be reached via free tiles.

    Performs a breadth-first search of the reachable free tiles until a target is encountered.
    If no target can be reached, the path that takes the agent closest to any target is chosen.

    Args:
        free_space: Boolean numpy array. True for free tiles and False for obstacles.
        start: the coordinate from which to begin the search.
        targets: list or array holding the coordinates of all target tiles.
        logger: optional logger object for debugging.
    Returns:
        coordinate of first step towards the closest target or towards tile closest to any target.
    """
    if len(targets) == 0:
        return None

    frontier = [start]
    parent_dict = {start: start}
    dist_so_far = {start: 0}
    best = start
    best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()

    while len(frontier) > 0:
        current = frontier.pop(0)
        # Find distance from current position to all targets, track closest
        d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
        if d + dist_so_far[current] <= best_dist:
            best = current
            best_dist = d + dist_so_far[current]
        if d == 0:
            # Found path to a target's exact position, mission accomplished!
            best = current
            break
        # Add unexplored free neighboring tiles to the queue in a random order
        x, y = current
        neighbors = [(x, y) for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] if free_space[x, y]]
        shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in parent_dict:
                frontier.append(neighbor)
                parent_dict[neighbor] = current
                dist_so_far[neighbor] = dist_so_far[current] + 1

    if logger: logger.debug(f'Suitable target found at {best}')
    # Determine the first step towards the best found target tile
    current = best
    while True:
        if parent_dict[current] == start: return current
        current = parent_dict[current]


def state_to_features(game_state: dict, coordinate_history: deque) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :param coordinate_history: A list of coordinates where agent has been in recent history, to detect loops
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    # Gather information about the game state. Normalize to -1 <= x <= 1.
    # Arena 17 x 17 = 289
    arena = game_state['field']

    # Step
    step = game_state['step'] / 400

    # Score, Bomb_avail, Coordinates, Alone
    score_self = game_state['self'][1] / 100
    bomb_avail = int(game_state['self'][2])
    self_x = game_state['self'][3][0]
    self_y = game_state['self'][3][1]
    self_x_normalized = self_x / 15
    self_y_normalized = self_y / 15
    try:
        score_opp1 = game_state['others'][0][1] / 100
        bomb_opp1 = int(game_state['others'][0][2])
        x_opp1 = game_state['others'][0][3][0] / 15
        y_opp1 = game_state['others'][0][3][1] / 15
        alone = 0
    except IndexError:
        score_opp1 = 0
        bomb_opp1 = 0
        x_opp1 = -1  # 0 or -1
        y_opp1 = -1
        alone = 1
    try:
        score_opp2 = game_state['others'][1][1] / 100
        bomb_opp2 = int(game_state['others'][1][2])
        x_opp2 = game_state['others'][1][3][0] / 15
        y_opp2 = game_state['others'][1][3][1] / 15
    except IndexError:
        score_opp2 = 0
        bomb_opp2 = 0
        x_opp2 = -1
        y_opp2 = -1
    try:
        score_opp3 = game_state['others'][2][1] / 100
        bomb_opp3 = int(game_state['others'][2][2])
        x_opp3 = game_state['others'][2][3][0] / 15
        y_opp3 = game_state['others'][2][3][1] / 15
    except IndexError:
        score_opp3 = 0
        bomb_opp3 = 0
        x_opp3 = -1
        y_opp3 = -1

    # In danger
    bomb_map = build_bomb_map(game_state)
    if bomb_map[self_x, self_y] == 5:
        in_danger = 0
    else:
        in_danger = 0.25 * (5 - bomb_map[self_x, self_y])

    # Placement
    opp_scores = [score_opp1, score_opp2, score_opp3]
    opp_scores.sort(reverse=True)
    if score_self > opp_scores[0]:
        placement = 1
    elif score_self > opp_scores[1]:
        placement = 0
    else:
        placement = -1

    # Up, Right, Down, Left, Touching_crate
    touching_crate = False
    up = tile_value(game_state, (self_x - 1, self_y), coordinate_history)
    right = tile_value(game_state, (self_x, self_y + 1), coordinate_history)
    down = tile_value(game_state, (self_x + 1, self_y), coordinate_history)
    left = tile_value(game_state, (self_x, self_y - 1), coordinate_history)
    if up == 0.5 or right == 0.5 or down == 0.5 or left == 0.5:
        touching_crate = True

    # Todo: shortest ways
    cols = range(1, arena.shape[0] - 1)
    rows = range(1, arena.shape[0] - 1)

    coins = game_state['coins']
    # dead_ends = [(x, y) for x in cols for y in rows if (arena[x, y] == 0)
    #              and ([arena[x + 1, y], arena[x - 1, y], arena[x, y + 1], arena[x, y - 1]].count(0) == 1)]
    crates = [(x, y) for x in cols for y in rows if (arena[x, y] == 1)]
    free_coins = coins
    free_crates = crates
    safe_tiles = [(x, y) for x in cols for y in rows if (arena[x, y] == 0)]

    # Exclude targets that are currently occupied by a bomb
    bombs = game_state['bombs']
    explosions = game_state['explosion_map']
    bomb_xys = [xy for (xy, t) in bombs]
    free_coins = [coin for coin in free_coins if bomb_map[coin[0], coin[1]] == 5 and explosions[coin[0], coin[1]] == 0]
    if len(free_coins) == 0:
        shortest_way_coin = "No More Coins"
    free_crates = [crate for crate in free_crates if bomb_map[crate[0], crate[1]] == 5 and explosions[crate[0], crate[1]] == 0]
    if len(free_crates) == 0:
        shortest_way_crate = "No More Crates"
    safe_tiles = [tile for tile in safe_tiles if bomb_map[tile[0], tile[1]] == 5 and explosions[tile[0], tile[1]] == 0]

    # Exclude tiles that are occupied by walls, crates, danger, explosions and others
    free_space = np.logical_and(arena == 0, explosions == 0, bomb_map == 5)
    others = [xy for (n, s, b, xy) in game_state['others']]
    for o in others:
        free_space[o] = False

    shortest_way_coin = 'No More Coins'
    dir_coin = look_for_targets(free_space, (self_x, self_y), free_coins)
    shortest_way_coin_up = 0.0
    shortest_way_coin_right = 0.0
    shortest_way_coin_down = 0.0
    shortest_way_coin_left = 0.0
    if dir_coin == (self_x - 1, self_y):
        shortest_way_coin_up = 1.0
        shortest_way_coin = "UP"
    if dir_coin == (self_x + 1, self_y):
        shortest_way_coin_down = 1.0
        shortest_way_coin = "DOWN"
    if dir_coin == (self_x, self_y - 1):
        shortest_way_coin_left = 1.0
        shortest_way_coin = "LEFT"
    if dir_coin == (self_x, self_y + 1):
        shortest_way_coin_right = 1.0
        shortest_way_coin = "RIGHT"

    print(shortest_way_coin_up, shortest_way_coin_right, shortest_way_coin_down, shortest_way_coin_left, shortest_way_coin)

    dir_crate = look_for_targets(free_space, (self_x, self_y), free_crates)
    shortest_way_crate_up = 0.0
    shortest_way_crate_right = 0.0
    shortest_way_crate_down = 0.0
    shortest_way_crate_left = 0.0
    if dir_crate == (self_x - 1, self_y):
        shortest_way_crate_up = 1.0
        shortest_way_crate = "UP"
    if dir_crate == (self_x + 1, self_y):
        shortest_way_crate_down = 1.0
        shortest_way_crate = "DOWN"
    if dir_crate == (self_x, self_y - 1):
        shortest_way_crate_left = 1.0
        shortest_way_crate = "LEFT"
    if dir_crate == (self_x, self_y + 1):
        shortest_way_crate_right = 1.0
        shortest_way_crate = "RIGHT"

    shortest_way_safety = 'Not In Danger'
    dir_safety = look_for_targets(free_space, (self_x, self_y), safe_tiles)
    shortest_way_safety_up = 0.0
    shortest_way_safety_right = 0.0
    shortest_way_safety_down = 0.0
    shortest_way_safety_left = 0.0
    if dir_safety == (self_x - 1, self_y):
        shortest_way_safety_up = 1.0
        shortest_way_safety = "UP"
    if dir_safety == (self_x + 1, self_y):
        shortest_way_safety_down = 1.0
        shortest_way_safety = "DOWN"
    if dir_safety == (self_x, self_y - 1):
        shortest_way_safety_left = 1.0
        shortest_way_safety = "LEFT"
    if dir_safety == (self_x, self_y + 1):
        shortest_way_safety_right = 1.0
        shortest_way_safety = "RIGHT"

    # Build feature vector
    flat_arena = arena.flatten()
    rest_features = np.array([step, score_self, bomb_avail, self_x_normalized, self_y_normalized,
                              score_opp1, score_opp2, score_opp3, bomb_opp1, bomb_opp2, bomb_opp3,
                              x_opp1, x_opp2, x_opp3, y_opp1, y_opp2, y_opp3, alone,
                              in_danger, placement, up, right, down, left, touching_crate,
                              shortest_way_coin_up, shortest_way_coin_right,
                              shortest_way_coin_down, shortest_way_coin_left,
                              shortest_way_crate_up, shortest_way_crate_right,
                              shortest_way_crate_down, shortest_way_crate_left])
    feature_vector = np.concatenate((flat_arena, rest_features), axis=0)

    return rest_features


state_to_features(experiment_state, coordinate_history)
print("<3")
