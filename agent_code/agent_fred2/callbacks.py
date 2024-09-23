import os
import pickle
import random
from collections import deque
import numpy as np
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .helpers import (look_for_targets, build_bomb_map, tile_value, coord_to_dir,
                      find_traps, best_explosion_score, explosion_score, passable, all_direction_distances,
                      guaranteed_passable_tiles, DIRECTIONS, bomb_explosion_map, is_safe
                      )
from .model import QNet

# if GPU is to be used
device = torch.device(
    'cuda' if torch.cuda.is_available() else
    'mps' if torch.backends.mps.is_available() else
    'cpu')

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
# up: -y
# right: +x
# down: +y
# left: -x

EPS_START = 0.5
EPS_END = 0.05
EPS_DECAY = 50

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
    self.steps = 0
    self.touching_crate = 0
    self.bomb_cooldown = 0

    if not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")

        self.model = QNet(22, 1024, 2048, 6)
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
    self.logger.debug(self.features)
    # self.logger.debug(game_state['explosion_map'].T)
    self.logger.debug(game_state['bombs'])

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

    features = []

    # Gather information about the game state. Normalize to -1 <= x <= 1.
    # Arena 17 x 17 = 289
    field = game_state['field']

    explosions = game_state['explosion_map']
    cols = range(1, field.shape[0] - 1)
    rows = range(1, field.shape[0] - 1)
    guaranteed_passable = guaranteed_passable_tiles(game_state)
    empty_tiles = [(x, y) for x in cols for y in rows if (field[x, y] == 0)]
    bomb_map = build_bomb_map(game_state)
    safe_tiles = [tile for tile in empty_tiles if bomb_map[tile[0], tile[1]] == 100 and \
                  explosions[tile[0], tile[1]] == 0]

    self.logger.debug(guaranteed_passable.T)

    # First step
    first_step = 0.0
    if game_state['step'] == 1:
        first_step = 1.0

    # Score, Bomb_avail, Coordinates, Alone
    score_self = game_state['self'][1] / 100
    bomb_avail = int(game_state['self'][2])
    self_x, self_y = game_state['self'][3]
    self_x_normalized = self_x / 16
    self_y_normalized = self_y / 16

    features.append(bomb_avail)
    features.append(self_x_normalized)
    features.append(self_y_normalized)

    # In danger
    if bomb_map[self_x, self_y] == 100:
        in_danger = 0.0
    else:
        in_danger = 1.0

    features.append(in_danger)

    # Do not place suicidal bombs
    bomb_explosion = bomb_explosion_map(game_state, self_x, self_y)
    if np.all(np.logical_or(bomb_explosion == 1.0, guaranteed_passable < 0)):
        suicidal_bomb = 1.0
    else:
        suicidal_bomb = 0.0

    # features.append(suicidal_bomb)

    # Distance to safety
    if in_danger == 1.0:
        passable_field = np.logical_and(guaranteed_passable >= 0, guaranteed_passable < 5)
        safety_distances = all_direction_distances(passable_field, (self_x, self_y), safe_tiles)
        if all(d == -1 for d in safety_distances):
            # In case there is no guaranteed safe tile, we can still try to reach one.
            passable_field = field == 0
            for other in game_state['others']:
                x, y = other[3]
                passable_field[x, y] = False
            for xy, t in game_state['bombs']:
                x, y = xy
                passable_field[x, y] = False
            safety_distances = all_direction_distances(passable_field, (self_x, self_y), safe_tiles)
        # Normalize to -1 <= x <= 1
        safety_distances = [1 - (d / 32) if d >= 0 else -1 for d in safety_distances]
    else:
        safety_distances = [0.0] * 4

    # +4 features
    features.extend(safety_distances)

    # Avoid repetetive movement
    tile_freq = [0.0] * 4
    for i, direction in enumerate(DIRECTIONS):
        x2, y2 = self_x + direction[0], self_y + direction[1]
        if not passable(x2, y2, game_state):
            continue
        tile_freq[i] = 1 / (self.coordinate_history.count((x2, y2)) + 1)
    tile_freq_stay = 1 / (self.coordinate_history.count((self_x, self_y)) + 1)

    # +5 features
    features.extend(tile_freq)
    features.append(tile_freq_stay)

    # Distance to coins
    coins = game_state['coins']
    passable_field = guaranteed_passable >= 0
    coin_distances = all_direction_distances(passable_field, (self_x, self_y), coins)
    if all(d == -1 for d in coin_distances):
        # In case there is no coin that is guaranteed to be reachable ignore opponent movement.
        temp_field = np.zeros_like(field)
        for x, y in safe_tiles:
            temp_field[x, y] = 1
        coin_distances = all_direction_distances(temp_field, (self_x, self_y), empty_tiles)
    # Normalize to -1 <= x <= 1
    coin_distances = [1 - (d / 32) if d >= 0 else -1 for d in coin_distances]

    # +4 features
    features.extend(coin_distances)

    # Avoid dangerous tiles
    safety = [0.0] * 4
    for i, direction in enumerate(DIRECTIONS):
        x2, y2 = self_x + direction[0], self_y + direction[1]
        if int(guaranteed_passable[x2, y2]) == 1 and is_safe(game_state, x2, y2):
            safety[i] = 1.0
    is_safe_stay = float(is_safe(game_state, self_x, self_y))

    # +5 features
    features.extend(safety)
    features.append(is_safe_stay)

    # TODO place good bombs

    return features
