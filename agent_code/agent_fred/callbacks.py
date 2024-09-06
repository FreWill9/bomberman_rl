import os
import pickle
import random
from collections import deque

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import matplotlib
import matplotlib.pyplot as plt

import pdb

from random import shuffle

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu")

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


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

    if not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")

        # weights = np.random.rand(len(ACTIONS))
        # self.model = weights / weights.sum()
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    self.coordinate_history.append((game_state['self'][3][0], game_state['self'][3][1]))

    state = state_to_features(game_state, self.coordinate_history)

    # todo Exploration vs exploitation
    random_prob = 0.0
    rounds = game_state['round']
    match rounds:
        case rounds if rounds < 60:
            random_prob = .2
        case rounds if 30 <= rounds < 200:
            random_prob = .1
        case rounds if 60 <= rounds < 600:
            random_prob = .05
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.25, .25, .25, .25, .0, .0])

    self.logger.debug("Querying model for action.")

    state0 = torch.tensor(state, dtype=torch.float)
    prediction = self.model(state0)

    move = torch.argmax(prediction).item()

    return ACTIONS[move]


def build_bomb_map(game_state: dict):
    bomb_map = np.ones(game_state['field'].shape) * 5
    for (xb, yb), t in game_state['bombs']:
        for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
            if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                bomb_map[i, j] = min(bomb_map[i, j], t)

    return bomb_map


def manhattan_dist(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)


def coin_dist_sum(game_state: dict, x, y):
    coins = game_state['coins']
    res = sum([2/(manhattan_dist(x, y, c[0], c[1] + 0.1)) for c in coins])
    # normalize
    return res / max(len(coins), 1)


def tile_value(game_state: dict, coord: (int, int), coordinate_history: deque) -> float:
    bomb_map = build_bomb_map(game_state)
    explosion_map = game_state['explosion_map']
    bomb_coord = [xy for (xy, t) in game_state['bombs']]
    opp_coord = [xy for (n, s, b, xy) in game_state['others']]
    value = 0.0

    match game_state['field'][coord[0], coord[1]]:
        case 0:
            pass
        case -1:
            return -1.0
        case 1:
            return 0.0
    if (coord[0], coord[1]) in opp_coord:
        return -1.0
    if bomb_map[coord[0], coord[1]] <= 1:
        return -1.0
    if explosion_map[coord[0], coord[1]] > 0:
        return -1.0
    if (coord[0], coord[1]) in bomb_coord:
        return -1.0
    if (coord[0], coord[1]) in game_state['coins']:
        return 1.0
    if coordinate_history.count((coord[0], coord[1])) > 2:
        value -= 0.3
    if bomb_map[coord[0], coord[1]] < 5:
        value -= 0.9

    value += coin_dist_sum(game_state, coord[0], coord[1])

    return value


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

    # Up, Right, Down, Left
    up = tile_value(game_state, (self_x - 1, self_y), coordinate_history)
    right = tile_value(game_state, (self_x, self_y + 1), coordinate_history)
    down = tile_value(game_state, (self_x + 1, self_y), coordinate_history)
    left = tile_value(game_state, (self_x, self_y - 1), coordinate_history)

    # Todo: shortest ways
    coins = game_state['coins']
    cols = range(1, arena.shape[0] - 1)
    rows = range(1, arena.shape[0] - 1)
    targets = coins
    free_space = arena == 0

    # Exclude targets that are currently occupied by a bomb
    bombs = game_state['bombs']
    bomb_xys = [xy for (xy, t) in bombs]
    targets = [target for target in targets if target not in bomb_xys]

    # Exclude free tiles that are occupied by others
    others = [xy for (n, s, b, xy) in game_state['others']]
    for o in others:
        free_space[o] = False

    d = look_for_targets(free_space, (self_x, self_y), targets)
    shortest_way_coin = 0.0
    if d == (self_x - 1, self_y):
        shortest_way_coin = 0.5
    if d == (self_x + 1, self_y):
        shortest_way_coin = -0.5
    if d == (self_x, self_y - 1):
        shortest_way_coin = -1.0
    if d == (self_x, self_y + 1):
        shortest_way_coin = 1

    # Build feature vector
    flat_arena = arena.flatten()
    rest_features = np.array([step, score_self, bomb_avail, self_x_normalized, self_y_normalized,
                              score_opp1, score_opp2, score_opp3, bomb_opp1, bomb_opp2, bomb_opp3,
                              x_opp1, x_opp2, x_opp3, y_opp1, y_opp2, y_opp3, alone,
                              in_danger, placement, up, right, down, left, shortest_way_coin])
    rest_features = np.array([up, right, down, left])
    feature_vector = np.concatenate((flat_arena, rest_features), axis=0)

    return rest_features


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
