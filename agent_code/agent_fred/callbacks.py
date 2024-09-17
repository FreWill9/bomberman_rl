import os
import pickle
import random
from collections import deque
from random import shuffle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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

    self.safety_stay = 1.0
    self.safety_up = 1.0
    self.safety_right = 1.0
    self.safety_down = 1.0
    self.safety_left = 1.0

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

    # todo Exploration vs exploitation
    random_prob = 0.0
    rounds = game_state['round']
    match rounds:
        case rounds if rounds < 10:
            random_prob = .5
        case rounds if 50 <= rounds < 100:
            random_prob = .2
        case rounds if 100 <= rounds < 200:
            random_prob = .05
        case rounds if 200 <= rounds < 600:
            random_prob = .01
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.20, .20, .20, .20, .20, .0])


    state = state_to_features(self, game_state)
    self.logger.info(f"Features: {state}")

    self.logger.debug("Querying model for action.")

    state = torch.tensor(state, dtype=torch.float)
    prediction = self.model(state).detach().numpy()

    # move = torch.argmax(prediction).item()
    prediction = (prediction-np.min(prediction))/(np.max(prediction)-np.min(prediction))
    pred_sum = sum(prediction)
    prediction = prediction / pred_sum
    move = np.random.choice(range(len(ACTIONS)), p=prediction)

    self.logger.debug(f"Predicted move: {ACTIONS[move]}.")

    return ACTIONS[move]


def build_bomb_map2(game_state: dict):
    bomb_map = np.ones(game_state['field'].shape) * 5
    for (xb, yb), t in game_state['bombs']:
        for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
            if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                bomb_map[i, j] = min(bomb_map[i, j], t)

    return bomb_map


def in_field(x, y, game_state):
    return 0 <= x < game_state['field'].shape[0] and 0 <= y < game_state['field'].shape[1]


def passable(x, y, game_state):
    return (in_field(x, y, game_state) and game_state['field'][x, y] == 0
            and (x, y) not in [xy for xy, t in game_state['bombs']]
            and (x, y) not in [xy for (n, s, b, xy) in game_state['others']])


def build_bomb_map(game_state: dict):
    bomb_map = np.ones(game_state['field'].shape) * 100
    for (xb, yb), t in game_state['bombs']:
        bomb_map[xb, yb] = min(bomb_map[xb, yb], t)
        for i in range(1, 4):
            if in_field(xb + i, yb, game_state) and game_state['field'][xb + i, yb] != -1:
                bomb_map[xb + i, yb] = min(bomb_map[xb + i, yb], t)
            else:
                break
        for i in range(1, 4):
            if in_field(xb - i, yb, game_state) and game_state['field'][xb - i, yb] != -1:
                bomb_map[xb - i, yb] = min(bomb_map[xb - i, yb], t)
            else:
                break
        for i in range(1, 4):
            if in_field(xb, yb + i, game_state) and game_state['field'][xb, yb + i] != -1:
                bomb_map[xb, yb + i] = min(bomb_map[xb, yb + i], t)
            else:
                break
        for i in range(1, 4):
            if in_field(xb, yb - i, game_state) and game_state['field'][xb, yb - i] != -1:
                bomb_map[xb, yb - i] = min(bomb_map[xb, yb - i], t)
            else:
                break

    return bomb_map


def safe_distance(game_state: dict, coord: (int, int), max_steps: int) -> int:
    """
    Calculate the distance to the closest safe tile from the given coordinate.
    """
    bomb_map = build_bomb_map(game_state)
    explosion_map = game_state['explosion_map'].copy()
    bombs = [(x, y) for (x, y), t in game_state['bombs']]

    if bomb_map[coord[0], coord[1]] == 100.0:
        return 0

    # Use BFS to find closest safe tile that is reachable (including waiting).
    tile_queue = deque([(coord[0], coord[1])])
    steps = 0
    while len(tile_queue) > 0 and steps <= max_steps:
        current_tile = tile_queue.popleft()
        x, y = current_tile
        if bomb_map[x, y] == 100.0:
            return steps

        if bomb_map[x, y] > 1.0 and explosion_map[x, y] == 0.0:
            tile_queue.append((x, y))

        if (game_state['field'][x + 1, y] == 0 and (x + 1, y) not in bombs
                and bomb_map[x + 1, y] > 1.0 and explosion_map[x + 1, y] == 0.0):
            tile_queue.append((x + 1, y))

        if (game_state['field'][x - 1, y] == 0 and (x - 1, y) not in bombs
                and bomb_map[x - 1, y] > 1.0 and explosion_map[x - 1, y] == 0.0):
            tile_queue.append((x - 1, y))

        if (game_state['field'][x, y + 1] == 0 and (x, y + 1) not in bombs
                and bomb_map[x, y + 1] > 1.0 and explosion_map[x, y + 1] == 0.0):
            tile_queue.append((x, y + 1))

        if (game_state['field'][x, y - 1] == 0 and (x, y - 1) not in bombs
                and bomb_map[x, y - 1] > 1.0 and explosion_map[x, y - 1] == 0.0):
            tile_queue.append((x, y - 1))

        steps += 1

        bomb_map = bomb_map - 1.0

        explosion_map = explosion_map + 1.0
        explosion_map[explosion_map != 2.0] = 0.0
        explosion_map[bomb_map == 0.0] = 1.0

        bomb_map[bomb_map <= -0.0] = 100.0

    return max_steps


def closest_coin_dist(game_state: dict, coord: (int, int)) -> int:
    """
    Calculate the distance to the closest coin from the coordinate using BFS.
    """
    coins = game_state['coins']
    if len(coins) == 0:
        return 0

    # Use BFS to find closest coin that is reachable.
    tile_queue = deque([(coord[0], coord[1], 0)])
    visited = np.zeros(game_state['field'].shape)
    visited[coord[0], coord[1]] = 1
    while len(tile_queue) > 0:
        x, y, step = tile_queue.popleft()
        if any([x == c[0] and y == c[1] for c in coins]):
            return step

        if passable(x + 1, y, game_state) and visited[x + 1, y] == 0:
            tile_queue.append((x + 1, y, step + 1))
            visited[x + 1, y] = 1

        if passable(x - 1, y, game_state) and visited[x - 1, y] == 0:
            tile_queue.append((x - 1, y, step + 1))
            visited[x - 1, y] = 1

        if passable(x, y + 1, game_state) and visited[x, y + 1] == 0:
            tile_queue.append((x, y + 1, step + 1))
            visited[x, y + 1] = 1

        if passable(x, y - 1, game_state) and visited[x, y - 1] == 0:
            tile_queue.append((x, y - 1, step + 1))
            visited[x, y - 1] = 1

    return 10000

def coin_score(game_state: dict, x, y) -> float:
    dist = closest_coin_dist(game_state, (x, y)) + 1
    if dist == 10000:
        return 0.0
    if dist >= 8:
        return 0.05
    return 1 - (dist / 8)

def manhattan_dist(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)


def coin_dist_sum(game_state: dict, x, y):
    coins = game_state['coins']
    res = sum([1 / (1 + manhattan_dist(x, y, c[0], c[1])) for c in coins])
    # normalize
    return res / max(len(coins), 1)


def tile_value(game_state: dict, coord: (int, int)) -> float:
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


def state_to_features(self, game_state: dict) -> np.array:
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
    arena = game_state['field']
    step = game_state['step'] / 400
    self_x, self_y = game_state['self'][3]

    # Can move
    up = float(passable(self_x, self_y - 1, game_state))
    right = float(passable(self_x + 1, self_y, game_state))
    down = float(passable(self_x, self_y + 1, game_state))
    left = float(passable(self_x - 1, self_y, game_state))

    def safety(val):
        if val == 0:
            return 1.0
        if val < 3:
            return 1.0
        if val == 3:
            return 1.0
        return 0.0

    # Tile safety
    self.safety_stay = safety(safe_distance(game_state, (self_x, self_y), 5))
    self.safety_up = safety(safe_distance(game_state, (self_x, self_y - 1), 5))
    self.safety_right = safety(safe_distance(game_state, (self_x + 1, self_y), 5))
    self.safety_down = safety(safe_distance(game_state, (self_x, self_y + 1), 5))
    self.safety_left = safety(safe_distance(game_state, (self_x - 1, self_y), 5))

    # Coins
    up_coins = coin_score(game_state, self_x, self_y - 1)
    right_coins = coin_score(game_state, self_x + 1, self_y)
    down_coins = coin_score(game_state, self_x, self_y + 1)
    left_coins = coin_score(game_state, self_x - 1, self_y)
    # best_val = max(up_coins, right_coins, down_coins, left_coins)
    # convert to binary for which one is best
    # up_coins = float(up_coins == best_val) - 0.0001
    # right_coins = float(right_coins == best_val) - 0.0002
    # down_coins = float(down_coins == best_val) - 0.0003
    # left_coins = float(left_coins == best_val)

    # Build feature vector
    features = np.array([up, right, down, left,
                         up_coins, right_coins, down_coins, left_coins,
                         #self.safety_stay, self.safety_up, self.safety_right, self.safety_down, self.safety_left,
                         ])

    return features


class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size1)
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)
        self.linear3 = nn.Linear(hidden_size2, hidden_size2)
        self.linear4 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        # x = F.relu(self.linear3(x))
        x = self.linear4(x)
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
