import os
import pickle
import random
from collections import deque
import numpy as np
import math
import copy
import torch

from .helpers import (look_for_targets, build_bomb_map, tile_value, coord_to_dir,
                      find_traps, best_explosion_score, explosion_score, closest_target_dist, safe_tile_reachable)
from .model import QNet

# if GPU is to be used
device = torch.device(
    'cuda' if torch.cuda.is_available() else
    'mps' if torch.backends.mps.is_available() else
    'cpu')
device = 'cpu'

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
# up: -y
# right: +x
# down: +y
# left: -x

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 25

FORCE_BOMBS = True


def setup(self):
    """
    Set up your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that is independent of the game state.

    :param self: This object is passed to all callbacks, and you can set arbitrary values.
    """

    self.coordinate_history = deque([], 20)
    self.shortest_way_coin = "None"
    self.shortest_way_crate = "None"
    self.shortest_way_safety = "None"
    self.shortest_way_trap = "None"
    self.steps = 0
    self.touching_crate = 0
    self.bomb_cooldown = 0

    if not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")

        self.model = QNet(20, 1024, 1024, 6)
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)

    self.logger.info(f"Using device: {device}")
    self.model.to(device)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    self.bomb_cooldown = max(0, self.bomb_cooldown - 1)
    self.features = state_to_features(self, game_state)
    self.step = game_state['step']
    self.x, self.y = game_state['self'][3]

    if self.step == 1:
        self.coordinate_history.clear()
    self.coordinate_history.append((self.x, self.y))

    action = choose_action(self, game_state)

    if action == 'BOMB' and self.bomb_cooldown <= 0:
        self.bomb_cooldown = 7

    return action


def choose_action(self, game_state: dict) -> str:
    if FORCE_BOMBS and game_state['step'] % 20 == 19 and self.bomb_cooldown <= 0:
        self.logger.debug("Force dropped bomb.")
        return 'BOMB'

    # Explore random actions with probability epsilon
    rounds_done = game_state['round']
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * rounds_done / EPS_DECAY)

    if self.train and random.random() <= eps_threshold:
        self.logger.debug(f"Choosing action purely at random. Prob: {eps_threshold * 100:.2f} %")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

    self.logger.debug("Querying model for action.")

    features = torch.tensor(self.features, dtype=torch.float).to(device)
    prediction = self.model(features)
    action = ACTIONS[torch.argmax(prediction).item()]

    self.logger.debug(f"Chose action {action}")

    return action


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
        alone = False
    except IndexError:
        score_opp1 = 0
        bomb_opp1 = 0
        x_opp1 = -1  # 0 or -1
        y_opp1 = -1
        alone = True
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

    explosion_scores = [float(i == best_explosion) for i in range(5)]

    # Assign shortest way coordinates to features
    if closest_target_dist(game_state, (self_x, self_y)) < 15 or alone:
        self.shortest_way_coin, shortest_way_coin_up, shortest_way_coin_right, \
            shortest_way_coin_down, shortest_way_coin_left = coord_to_dir(self_x, self_y, dir_coin)

    if best_explosion == -1:
        crate_dirs = coord_to_dir(self_x, self_y, dir_crate)
        self.shortest_way_crate = crate_dirs[0]
        explosion_scores = list(crate_dirs[1:5]) + [0.0]

    (explosion_score_up, explosion_score_right,
     explosion_score_down, explosion_score_left, explosion_score_stay) = explosion_scores

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

    test_vector = np.array([in_danger, bomb_avail, self.touching_crate, explosion_score_stay,
                            shortest_way_coin_up, shortest_way_coin_right,
                            shortest_way_coin_down, shortest_way_coin_left,
                            shortest_way_safety_up, shortest_way_safety_right,
                            shortest_way_safety_down, shortest_way_safety_left,
                            shortest_way_opp_up, shortest_way_opp_right,
                            shortest_way_opp_down, shortest_way_opp_left,
                            explosion_score_up, explosion_score_right,
                            explosion_score_down, explosion_score_left])

    # For debugging
    self.logger.debug(f"\n"
                      f"Proposed way coin: {self.shortest_way_coin} \n"
                      f"Proposed way crate: {self.shortest_way_crate} \n"
                      f"Proposed way safety: {self.shortest_way_safety} \n"
                      f"Proposed way opp: {self.shortest_way_opp}"
                      f"tile values: {up, right, down, left} \n")

    return test_vector
