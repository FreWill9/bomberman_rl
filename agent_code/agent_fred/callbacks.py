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

from .helpers import look_for_targets, build_bomb_map, tile_value

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu")

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 600


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
    self.shortest_way_coin = "No More Coins"
    self.shortest_way_crate = "No More Crates"
    self.shortest_way_safety = "Not In Danger"
    self.steps = 0
    self.counter = 0

    if not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")

        # weights = np.random.rand(len(ACTIONS))
        # self.model = weights / weights.sum()
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)


def act(self, game_state: dict) -> str:
    if game_state['step'] % 20 == 19:
        self.logger.debug("Forced drop Bomb")
        return 'BOMB'
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    self.coordinate_history.append((game_state['self'][3][0], game_state['self'][3][1]))
    self.steps = game_state['step']

    state = state_to_features(self, game_state, self.coordinate_history)

    # todo Exploration vs exploitation
    rounds_done = game_state['round']
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * rounds_done / EPS_DECAY)
    sample = random.random()

    if self.train and sample <= eps_threshold:
        with torch.no_grad():
            self.logger.debug(f"Choosing action purely at random. Prob: {eps_threshold}")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .2, .0])

    self.logger.debug("Querying model for action.")

    state0 = torch.tensor(state, dtype=torch.float)
    prediction = self.model(state0)
    move = torch.argmax(prediction).item()

    self.logger.debug(f"Chose action {ACTIONS[move]}")

    return ACTIONS[move]


def state_to_features(self, game_state: dict, coordinate_history: deque) -> np.array:
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
        self.shortest_way_safety = "Not In Danger"
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
        self.shortest_way_coin = "No More Coins"
    free_crates = [crate for crate in free_crates if bomb_map[crate[0], crate[1]] == 5 and explosions[crate[0], crate[1]] == 0]
    if len(free_crates) == 0:
        self.shortest_way_crate = "No More Crates"
    safe_tiles = [tile for tile in safe_tiles if bomb_map[tile[0], tile[1]] == 5 and explosions[tile[0], tile[1]] == 0]

    # Exclude tiles that are occupied by walls, crates, danger, explosions and others
    free_space = np.logical_and(arena == 0, explosions == 0, bomb_map == 5)
    others = [xy for (n, s, b, xy) in game_state['others']]
    for o in others:
        free_space[o] = False

    dir_coin = look_for_targets(free_space, (self_x, self_y), free_coins)
    shortest_way_coin_up = 0.0
    shortest_way_coin_right = 0.0
    shortest_way_coin_down = 0.0
    shortest_way_coin_left = 0.0
    if dir_coin == (self_x - 1, self_y):
        shortest_way_coin_up = 1.0
        self.shortest_way_coin = "UP"
    if dir_coin == (self_x + 1, self_y):
        shortest_way_coin_down = 1.0
        self.shortest_way_coin = "DOWN"
    if dir_coin == (self_x, self_y - 1):
        shortest_way_coin_left = 1.0
        self.shortest_way_coin = "LEFT"
    if dir_coin == (self_x, self_y + 1):
        shortest_way_coin_right = 1.0
        self.shortest_way_coin = "RIGHT"

    dir_crate = look_for_targets(free_space, (self_x, self_y), free_crates)
    shortest_way_crate_up = 0.0
    shortest_way_crate_right = 0.0
    shortest_way_crate_down = 0.0
    shortest_way_crate_left = 0.0
    if dir_crate == (self_x - 1, self_y):
        shortest_way_crate_up = 1.0
        self.shortest_way_crate = "UP"
    if dir_coin == (self_x + 1, self_y):
        shortest_way_crate_down = 1.0
        self.shortest_way_crate = "DOWN"
    if dir_coin == (self_x, self_y - 1):
        shortest_way_crate_left = 1.0
        self.shortest_way_crate = "LEFT"
    if dir_coin == (self_x, self_y + 1):
        shortest_way_crate_right = 1.0
        self.shortest_way_crate = "RIGHT"

    dir_safety = look_for_targets(free_space, (self_x, self_y), safe_tiles)
    shortest_way_safety_up = 0.0
    shortest_way_safety_right = 0.0
    shortest_way_safety_down = 0.0
    shortest_way_safety_left = 0.0
    if dir_safety == (self_x - 1, self_y):
        shortest_way_safety_up = 1.0
        self.shortest_way_safety = "UP"
    if dir_safety == (self_x + 1, self_y):
        shortest_way_safety_down = 1.0
        self.shortest_way_safety = "DOWN"
    if dir_safety == (self_x, self_y - 1):
        shortest_way_safety_left = 1.0
        self.shortest_way_safety = "LEFT"
    if dir_safety == (self_x, self_y + 1):
        shortest_way_safety_right = 1.0
        self.shortest_way_safety = "RIGHT"

    # Build feature vector
    flat_arena = arena.flatten()
    rest_features = np.array([step, score_self, bomb_avail, self_x_normalized, self_y_normalized,
                              score_opp1, score_opp2, score_opp3, bomb_opp1, bomb_opp2, bomb_opp3,
                              x_opp1, x_opp2, x_opp3, y_opp1, y_opp2, y_opp3, alone,
                              in_danger, placement, up, right, down, left, touching_crate,
                              shortest_way_coin_up, shortest_way_coin_right,
                              shortest_way_coin_down, shortest_way_coin_left,
                              shortest_way_crate_up, shortest_way_crate_right,
                              shortest_way_crate_down, shortest_way_crate_left,
                              shortest_way_safety_up, shortest_way_safety_right,
                              shortest_way_safety_down, shortest_way_safety_left])
    feature_vector = np.concatenate((flat_arena, rest_features), axis=0)

    test_vector = np.array([in_danger, shortest_way_coin_up, shortest_way_coin_right,
                            shortest_way_coin_down, shortest_way_coin_left,
                            shortest_way_safety_up, shortest_way_safety_right,
                            shortest_way_safety_down, shortest_way_safety_left])

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
