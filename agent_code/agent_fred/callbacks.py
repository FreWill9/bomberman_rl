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

from .helpers import look_for_targets, build_bomb_map, tile_value, coord_to_dir, transpose_action

# if GPU is to be used
device = torch.device(
    'cuda' if torch.cuda.is_available() else
    'mps' if torch.backends.mps.is_available() else
    'cpu')

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 25


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
    self.counter = 0
    self.touching_crate = 0

    if not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")

        self.model = Linear_QNet(20, 512, 512, 6).to(device)
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)


def act(self, game_state: dict) -> str:

    """if game_state['step'] % 20 == 19:
        self.logger.debug("Forced drop Bomb")
        return 'BOMB'"""

    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    self.state = state_to_features(self, game_state)

    self.step = game_state['step']
    if self.step == 1:
        self.coordinate_history.clear()
    self.coordinate_history.append((game_state['self'][3][0], game_state['self'][3][1]))

    # Exploration vs exploitation
    rounds_done = game_state['round']
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * rounds_done / EPS_DECAY)
    sample = random.random()

    if self.train and sample <= eps_threshold:
        with torch.no_grad():
            self.logger.debug(f"Choosing action purely at random. Prob: {int(eps_threshold * 100)} %")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

    self.logger.debug("Querying model for action.")

    state0 = torch.tensor(self.state, dtype=torch.float)
    prediction = self.model(state0)
    move = torch.argmax(prediction).item()

    self.logger.debug(f"Chose action {ACTIONS[move]}")

    return ACTIONS[move]


def state_to_features(self, game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

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
    opp_scores = [score_opp1, score_opp2, score_opp3]
    opp_scores.sort(reverse=True)
    if score_self > opp_scores[0]:
        placement = 1
    elif score_self > opp_scores[1]:
        placement = 0
    else:
        placement = -1

    # Up, Right, Down, Left, Touching_crate
    self.touching_crate = 0
    up = tile_value(game_state, (self_x - 1, self_y), self.coordinate_history)
    right = tile_value(game_state, (self_x, self_y + 1), self.coordinate_history)
    down = tile_value(game_state, (self_x + 1, self_y), self.coordinate_history)
    left = tile_value(game_state, (self_x, self_y - 1), self.coordinate_history)
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

    # Exclude targets that are currently occupied by bomb or explosion
    free_coins = [coin for coin in coins if bomb_map[coin[0], coin[1]] == 100 and \
                  explosions[coin[0], coin[1]] == 0]

    free_crates = [crate for crate in crates if bomb_map[crate[0], crate[1]] == 100 and \
                   explosions[crate[0], crate[1]] == 0]

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

    others = [xy for (n, s, b, xy) in game_state['others']]
    for o in others:
        free_space[o] = False
        escape_space[o] = False

    # Compute shortest way coordinates
    dir_coin = look_for_targets(free_space, (self_x, self_y), free_coins)
    dir_crate = look_for_targets(free_space, (self_x, self_y), free_crates)
    dir_safety = look_for_targets(escape_space, (self_x, self_y), safe_tiles)

    # Assign shortest way coordinates to features
    if in_danger == 0.0:
        self.shortest_way_coin, shortest_way_coin_up, shortest_way_coin_right, \
            shortest_way_coin_down, shortest_way_coin_left = coord_to_dir(self_x, self_y, dir_coin)

    if in_danger == 0.0 and self.shortest_way_coin == "None" and self.touching_crate == 0:
        self.shortest_way_crate, shortest_way_crate_up, shortest_way_crate_right, \
            shortest_way_crate_down, shortest_way_crate_left = coord_to_dir(self_x, self_y, dir_crate)

    if in_danger != 0.0:
        self.shortest_way_safety, shortest_way_safety_up, shortest_way_safety_right, \
            shortest_way_safety_down, shortest_way_safety_left = coord_to_dir(self_x, self_y, dir_safety)

    # Build feature vector
    flat_arena = arena.flatten()
    rest_features = np.array([first_step, score_self, bomb_avail, self_x_normalized, self_y_normalized,
                              score_opp1, score_opp2, score_opp3, bomb_opp1, bomb_opp2, bomb_opp3,
                              x_opp1, x_opp2, x_opp3, y_opp1, y_opp2, y_opp3, alone,
                              in_danger, placement, up, right, down, left, self.touching_crate,
                              shortest_way_coin_up, shortest_way_coin_right,
                              shortest_way_coin_down, shortest_way_coin_left,
                              shortest_way_crate_up, shortest_way_crate_right,
                              shortest_way_crate_down, shortest_way_crate_left,
                              shortest_way_safety_up, shortest_way_safety_right,
                              shortest_way_safety_down, shortest_way_safety_left])
    feature_vector = np.concatenate((flat_arena, rest_features), axis=0)

    test_vector = np.array([in_danger, bomb_avail, up, right, down, left, self.touching_crate, first_step,
                            shortest_way_coin_up, shortest_way_coin_right,
                            shortest_way_coin_down, shortest_way_coin_left,
                            shortest_way_crate_up, shortest_way_crate_right,
                            shortest_way_crate_down, shortest_way_crate_left,
                            shortest_way_safety_up, shortest_way_safety_right,
                            shortest_way_safety_down, shortest_way_safety_left])

    # For debugging
    self.logger.debug(f"\n"
                      f"Proposed way coin: {transpose_action(self.shortest_way_coin)} \n"
                      f"Proposed way crate: {transpose_action(self.shortest_way_crate)} \n"
                      f"Proposed way safety: {transpose_action(self.shortest_way_safety)} \n")

    return test_vector


class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size1)
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)
        self.linear3 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.SmoothL1Loss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float).to(device)
        action = torch.tensor(action, dtype=torch.float).to(device)
        reward = torch.tensor(reward, dtype=torch.float).to(device)
        next_state = torch.tensor(next_state, dtype=torch.float).to(device)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        # 1: predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()

        for idx in range(len(done)):
            q_new = reward[idx]
            if not done[idx]:
                q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx][int(action[idx].item())] = q_new

        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(pred, target)
        loss.backward()

        self.optimizer.step()
