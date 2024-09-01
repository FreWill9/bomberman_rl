import numpy as np
import torch
from collections import namedtuple, deque
import random

experiment_state = {'round': 1,

                    'step': 9,

                    'field': np.array([[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                                       [-1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, -1],
                                       [-1, 0, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 0, -1],
                                       [-1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, -1],
                                       [-1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
                                       [-1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, -1],
                                       [-1, 0, -1, 1, -1, 0, -1, 0, -1, 0, -1, 1, -1, 1, -1, 1, -1],
                                       [-1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, -1],
                                       [-1, 1, -1, 1, -1, 0, -1, 1, -1, 1, -1, 1, -1, 1, -1, 0, -1],
                                       [-1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, -1],
                                       [-1, 1, -1, 1, -1, 0, -1, 1, -1, 0, -1, 1, -1, 0, -1, 1, -1],
                                       [-1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, -1],
                                       [-1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 0, -1, 1, -1],
                                       [-1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, -1],
                                       [-1, 0, -1, 1, -1, 0, -1, 1, -1, 1, -1, 0, -1, 1, -1, 0, -1],
                                       [-1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, -1],
                                       [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]),

                    'self': ('random_agent', 0, True, (np.int64(15), np.int64(1))),

                    'others': [  # ('peaceful_agent', 0, True, (np.int64(2), np.int64(15))),
                        # ('coin_collector_agent', 0, False, (np.int64(1), np.int64(1)))
                    ],

                    'bombs': [((np.int64(11), np.int64(14)), 1),
                              ((np.int64(1), np.int64(2)), 2)],

                    'coins': [(np.int64(7), np.int64(13)),
                              (np.int64(3), np.int64(9))],

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


def state_to_features(game_state: dict) -> np.array:
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

    # Gather information about the game state in multiple maps
    # Walls and Crates map
    walls_crates_map = game_state['field']

    # Bomb map
    bombs = game_state['bombs']
    bomb_xys = [xy for (xy, t) in bombs]
    bomb_map = np.ones(walls_crates_map.shape) * 5
    for (xb, yb), t in bombs:
        for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
            if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                bomb_map[i, j] = min(bomb_map[i, j], t)

    # Self map Todo: Bomb available?
    self_name, self_score, self_bomb_avail, (self_x, self_y) = game_state['self']
    self_map = np.ones(walls_crates_map.shape) * -1
    self_map[self_x, self_y] = self_score

    # Coin map
    coin_map = np.zeros(walls_crates_map.shape)
    coins = game_state['coins']
    for (coin_x, coin_y) in coins:
        coin_map[coin_x, coin_y] = 1

    # Others map Todo: Bombs available?
    others_score = [s for (n, s, b, xy) in game_state['others']]
    if len(others_score) > 0:
        (others_x, others_y) = [(x, y) for (n, s, b, (x, y)) in game_state['others']]
    others_bomb_avail = [b for (n, s, b, xy) in game_state['others']]
    others_map = np.ones(walls_crates_map.shape) * -1
    for i in range(len(others_score)):
        others_map[others_x[i], others_y[i]] = others_score[i]

    # Explosion map
    explosion_map = game_state['explosion_map']

    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    # For example, you could construct several channels of equal shape, ...
    channels = [walls_crates_map, bomb_map, self_map, coin_map, others_map, explosion_map]
    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels)
    # and return them as a vector
    return stacked_channels.reshape(-1)


def encode_action(action):
    match action:
        case 'UP':
            return 1.0
        case 'RIGHT':
            return 2.0
        case 'DOWN':
            return 3.0
        case 'LEFT':
            return 4.0
        case 'WAIT':
            return 5.0
        case 'BOMB':
            return 6.0


BATCH_SIZE = 1000
MAX_MEMORY = 100_000

memory = deque(maxlen=MAX_MEMORY)

state = state_to_features(experiment_state)

# remember
memory.append(
    Memory(state_to_features(experiment_state), 'UP', 10, None, True))

memory.append(
    Memory(state_to_features(experiment_state), 'UP', 10, None, True))

# train long memory
if len(memory) > BATCH_SIZE:
    mini_sample = random.sample(memory, BATCH_SIZE)  # list of tuples
else:
    mini_sample = memory

states, actions, rewards, next_states, dones = zip(*mini_sample)

state = torch.tensor(states, dtype=torch.float)
print(state.shape)
print(f"state shape: {state.shape}, len state shape: {len(state.shape)}")
print("<3")
