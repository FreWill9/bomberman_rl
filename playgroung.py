import numpy as np
import copy
from collections import namedtuple, deque
from agent_code.agent_fred.helpers import (build_bomb_map, tile_value, look_for_targets,
                                           coord_to_dir, find_traps, explosion_score,
                                           closest_target_dist, best_explosion_score, safe_tile_reachable)

experiment_state = {'round': 1,

                    'step': 9,

                    'field': np.array(
                        [[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                         [-1, 0,  0, 0,  0, 0,  1, 0,  0, 0,  0, 0,  0, 0,  0, 0, -1],
                         [-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1],
                         [-1, 0,  0, 0,  0, 0,  0, 0,  0, 0,  0, 0,  0, 0,  0, 0, -1],
                         [-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1],
                         [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                         [-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1],
                         [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                         [-1, 0, 0, 0, -1, 0, -1, 1, -1, 0, -1, 0, -1, 0, -1, 0, -1],
                         [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                         [-1, 0, 0, 0, 0, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1],
                         [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                         [-1, 0, 0, -1, 0, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1],
                         [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                         [-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1],
                         [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                         [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]),

                    'self': ('test', 2, False, (np.int64(3), np.int64(1))),

                    'others': [  # ("ooo", 2, True, (np.int64(3), np.int64(7))),
                               # ("oo", 2, True, (np.int64(15), np.int64(14))),
                               ],

                    'bombs': [],

                    'coins': [],

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


class Self:
    def __init__(self):
        self.shortest_way_coin = "None"
        self.shortest_way_crate = "None"
        self.shortest_way_safety = "None"
        self.shortest_way_trap = "None"
        self.touching_crate = 0
        self.coordinate_history = deque([], 20)
        self.placement = -5
        self.bomb_cooldown = 0
        self.shortest_way_opp = "None"
        self.trap_tiles = []
        self.bomb_for_trap_tiles = []


self = Self()

Memory = namedtuple('Memory',
                    ('state', 'action', 'reward', 'next_state', 'done'))

coordinate_history = deque([(3, 3), (2, 2), (3, 3), (4, 2), (3, 3)], 20)

ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT"]


def state_to_features(self, game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    # Gather information about the game state. Normalize to -1 <= x <= 1.
    # Arena 17 x 17 = 289
    arena = game_state['field']

    # First step
    first_step = 0.0
    if game_state['step'] == 1:
        first_step = 1.0

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
    if bomb_map[self_x, self_y] == 100:
        in_danger = 0.0
        self.shortest_way_safety = "None"
    else:
        in_danger = 1.0

    # Placement
    all_scores = [score_self, score_opp1, score_opp2, score_opp3]
    all_scores.sort(reverse=True)
    self.placement = all_scores.index(score_self) + 1

    # Up, Right, Down, Left, Touching_crate
    self.touching_crate = 0
    up = tile_value(game_state, (self_x, self_y - 1), self.coordinate_history)
    right = tile_value(game_state, (self_x + 1, self_y), self.coordinate_history)
    down = tile_value(game_state, (self_x, self_y + 1), self.coordinate_history)
    left = tile_value(game_state, (self_x - 1, self_y), self.coordinate_history)
    if arena[self_x - 1, self_y] == 1 or arena[self_x + 1, self_y] == 1 or \
            arena[self_x, self_y - 1] == 1 or arena[self_x, self_y + 1] == 1:
        self.touching_crate = 1

    # Shortest ways
    # Initialize with zero:
    shortest_way_coin_up = 0.0
    shortest_way_coin_right = 0.0
    shortest_way_coin_down = 0.0
    shortest_way_coin_left = 0.0
    self.shortest_way_coin = 'None'

    shortest_way_crate_up = 0.0
    shortest_way_crate_right = 0.0
    shortest_way_crate_down = 0.0
    shortest_way_crate_left = 0.0
    self.shortest_way_crate = 'None'

    shortest_way_opp_up = 0.0
    shortest_way_opp_right = 0.0
    shortest_way_opp_down = 0.0
    shortest_way_opp_left = 0.0
    self.shortest_way_opp = 'None'

    shortest_way_safety_up = 0.0
    shortest_way_safety_right = 0.0
    shortest_way_safety_down = 0.0
    shortest_way_safety_left = 0.0
    self.shortest_way_safety = 'None'

    explosions = game_state['explosion_map']
    coins = game_state['coins']

    cols = range(1, arena.shape[0] - 1)
    rows = range(1, arena.shape[0] - 1)

    crates = [(x, y) for x in cols for y in rows if (arena[x, y] == 1)]
    empty_tiles = [(x, y) for x in cols for y in rows if (arena[x, y] == 0)]
    bomb_xys = [xy for (xy, t) in game_state['bombs']]
    others = [xy for (n, s, b, xy) in game_state['others']]
    self.trap_tiles, self.bomb_for_trap_tiles = find_traps(game_state)
    trap_tiles = self.trap_tiles
    bomb_for_trap_tiles = self.bomb_for_trap_tiles

    # Exclude targets that are currently occupied by bomb or explosion
    free_coins = [coin for coin in coins if bomb_map[coin[0], coin[1]] == 100 and \
                  explosions[coin[0], coin[1]] == 0 and (coin[0], coin[1]) not in trap_tiles]

    free_crates = [crate for crate in crates if bomb_map[crate[0], crate[1]] == 100 and \
                   explosions[crate[0], crate[1]] == 0 and (crate[0], crate[1]) not in trap_tiles]

    free_opps = [opp for opp in others if bomb_map[opp[0], opp[1]] == 100 and \
                 explosions[opp[0], opp[1]] == 0 and (opp[0], opp[1]) not in trap_tiles]

    escape_tiles = [tile for tile in empty_tiles if explosions[tile[0], tile[1]] == 0 and tile not in bomb_xys]

    safe_tiles = [tile for tile in empty_tiles if bomb_map[tile[0], tile[1]] == 100 and \
                  explosions[tile[0], tile[1]] == 0]

    # Exclude ways that are occupied by walls, crates, danger, explosions and others
    free_space = np.zeros((arena.shape[0], arena.shape[1]), dtype=bool)
    for tile in safe_tiles:
        free_space[tile[0], tile[1]] = True

    escape_space = np.zeros((arena.shape[0], arena.shape[1]), dtype=bool)
    for tile in escape_tiles:
        escape_space[tile[0], tile[1]] = True

    for o in others:
        free_space[o] = False
        escape_space[o] = False

    for t in trap_tiles:
        free_space[t] = False

    # Compute shortest way coordinates
    dir_coin = look_for_targets(free_space, (self_x, self_y), free_coins)
    dir_crate = look_for_targets(free_space, (self_x, self_y), free_crates)
    dir_opp = look_for_targets(free_space, (self_x, self_y), free_opps)
    dir_safety = look_for_targets(escape_space, (self_x, self_y), safe_tiles)

    # find best explosion direction
    max_steps = self.bomb_cooldown + 5
    explosion_score_up = best_explosion_score(self, game_state, (self_x, self_y), (0, -1), max_steps)
    explosion_score_right = best_explosion_score(self, game_state, (self_x, self_y), (1, 0), max_steps)
    explosion_score_down = best_explosion_score(self, game_state, (self_x, self_y), (0, 1), max_steps)
    explosion_score_left = best_explosion_score(self, game_state, (self_x, self_y), (-1, 0), max_steps)
    explosion_score_stay = explosion_score(self, game_state, self_x, self_y)

    explosion_scores = [explosion_score_up, explosion_score_right, explosion_score_down, explosion_score_left,
                        explosion_score_stay]

    print(np.argmax([1, 2, 2, 1]))

    best_explosion = np.argmax(explosion_scores[:4])
    pot_game_state = copy.deepcopy(game_state)
    pot_game_state['bombs'].append(((self_x, self_y), 5))
    if explosion_scores[best_explosion] == 0 or not safe_tile_reachable(pot_game_state, (self_x, self_y)):
        best_explosion = -1
        self.shortest_way_crate = "None"
    elif explosion_scores[4] >= explosion_scores[best_explosion] and game_state['self'][2]:
        best_explosion = 4
        self.shortest_way_crate = "BOMB"
    else:
        self.shortest_way_crate = ACTIONS[best_explosion]
    print(explosion_scores)
    explosion_scores = [float(i == best_explosion) for i in range(5)]
    print(explosion_scores)

    # Assign shortest way coordinates to features
    if closest_target_dist(game_state, (self_x, self_y)) < 15:
        self.shortest_way_coin, shortest_way_coin_up, shortest_way_coin_right, \
            shortest_way_coin_down, shortest_way_coin_left = coord_to_dir(self_x, self_y, dir_coin)

    if best_explosion == -1:
        crate_dirs = coord_to_dir(self_x, self_y, dir_crate)
        self.shortest_way_crate = crate_dirs[0]
        explosion_scores = list(crate_dirs[1:5]) + [0.0]

    self.shortest_way_opp, shortest_way_opp_up, shortest_way_opp_right, \
        shortest_way_opp_down, shortest_way_opp_left = coord_to_dir(self_x, self_y, dir_opp)

    if in_danger != 0.0:
        self.shortest_way_safety, shortest_way_safety_up, shortest_way_safety_right, \
            shortest_way_safety_down, shortest_way_safety_left = coord_to_dir(self_x, self_y, dir_safety)

    # Build feature vector
    flat_arena = arena.flatten()
    rest_features = np.array([first_step, score_self, bomb_avail, self_x_normalized, self_y_normalized,
                              score_opp1, score_opp2, score_opp3, bomb_opp1, bomb_opp2, bomb_opp3,
                              x_opp1, x_opp2, x_opp3, y_opp1, y_opp2, y_opp3, alone,
                              in_danger, self.placement / 4, up, right, down, left, self.touching_crate,
                              shortest_way_coin_up, shortest_way_coin_right,
                              shortest_way_coin_down, shortest_way_coin_left,
                              # shortest_way_crate_up, shortest_way_crate_right,
                              # shortest_way_crate_down, shortest_way_crate_left,
                              shortest_way_safety_up, shortest_way_safety_right,
                              shortest_way_safety_down, shortest_way_safety_left])
    feature_vector = np.concatenate((flat_arena, rest_features), axis=0)

    test_vector = np.array([in_danger, bomb_avail, up, right, down, left,
                            self.touching_crate, first_step, 0,
                            shortest_way_coin_up, shortest_way_coin_right,
                            shortest_way_coin_down, shortest_way_coin_left,
                            shortest_way_safety_up, shortest_way_safety_right,
                            shortest_way_safety_down, shortest_way_safety_left,
                            shortest_way_opp_up, shortest_way_opp_right,
                            shortest_way_opp_down, shortest_way_opp_left,
                            *explosion_scores])

    # For debugging
    print(f"\n"
          f"{np.transpose(arena)} \n"
          f"self: {(self_y, self_x)} \n"
          f"Proposed way coin: {self.shortest_way_coin} \n"
          f"Proposed way crate: {self.shortest_way_crate} \n"
          f"Proposed way safety: {self.shortest_way_safety} \n"
          f"Proposed way opp: {self.shortest_way_opp} \n")

    return test_vector


features = state_to_features(self, experiment_state)

print("<3")
