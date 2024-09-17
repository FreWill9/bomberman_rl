import numpy as np
from random import shuffle
from collections import deque
from IPython import display
import matplotlib.pyplot as plt
import copy


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
        # shuffle(neighbors)
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
    if bomb_map[coord[0], coord[1]] != 100:
        temp = -0.9
    if explosion_map[coord[0], coord[1]] > 0:
        temp = -1.0
    if (coord[0], coord[1]) in bomb_coord:
        temp = -1.0

    return temp


def in_field(x, y, game_state) -> bool:
    return 0 <= x < game_state['field'].shape[0] and 0 <= y < game_state['field'].shape[1]


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


def transpose_action(action: str) -> str:
    match action:
        case "UP":
            return "LEFT"
        case "RIGHT":
            return "DOWN"
        case "DOWN":
            return "RIGHT"
        case "LEFT":
            return "UP"
        case _:
            return action


def encode_action(action: str) -> float:
    match action:
        case 'UP':
            return 0.0
        case 'RIGHT':
            return 1.0
        case 'DOWN':
            return 2.0
        case 'LEFT':
            return 3.0
        case 'WAIT':
            return 4.0
        case 'BOMB':
            return 5.0


def coord_to_dir(x, y, coord_target) -> (str, float, float, float, float):
    if coord_target is None or (x, y) == coord_target:
        return 'None', 0.0, 0.0, 0.0, 0.0
    if coord_target == (x - 1, y):
        return 'UP', 1.0, 0.0, 0.0, 0.0
    if coord_target == (x, y + 1):
        return 'RIGHT', 0.0, 1.0, 0.0, 0.0
    if coord_target == (x + 1, y):
        return 'DOWN', 0.0, 0.0, 1.0, 0.0
    if coord_target == (x, y - 1):
        return 'LEFT', 0.0, 0.0, 0.0, 1.0


def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)


def mirror_game_state(game_state: dict) -> (dict, dict, dict):
    x = copy.deepcopy(game_state)
    y = copy.deepcopy(game_state)
    xy = copy.deepcopy(game_state)
    size_x = game_state['field'].shape[0]
    size_y = game_state['field'].shape[1]

    x['field'] = np.flipud(game_state['field'])
    y['field'] = np.fliplr(game_state['field'])
    xy['field'] = np.flipud(np.fliplr(game_state['field']))

    for i in range(len(game_state['bombs'])):
        x['bombs'][i] = ((abs(size_x - 1 - game_state['bombs'][i][0][0]), game_state['bombs'][i][0][1]),
                         game_state['bombs'][i][1])
        y['bombs'][i] = ((game_state['bombs'][i][0][0], abs(size_y - 1 - game_state['bombs'][i][0][1])),
                         game_state['bombs'][i][1])
        xy['bombs'][i] = ((abs(size_x - 1 - game_state['bombs'][i][0][0]),
                           abs(size_y - 1 - game_state['bombs'][i][0][1])),
                          game_state['bombs'][i][1])

    x['explosion_map'] = np.flipud(game_state['explosion_map'])
    y['explosion_map'] = np.fliplr(game_state['explosion_map'])
    xy['explosion_map'] = np.flipud(np.fliplr(game_state['explosion_map']))

    for i in range(len(game_state['coins'])):
        x['coins'][i] = (abs(size_x - 1 - game_state['coins'][i][0]), game_state['coins'][i][1])
        y['coins'][i] = (game_state['coins'][i][0], abs(size_y - 1 - game_state['coins'][i][1]))
        xy['coins'][i] = (abs(size_x - 1 - game_state['coins'][i][0]), abs(size_y - 1 - game_state['coins'][i][1]))

    x['self'] = (game_state['self'][0], game_state['self'][1], game_state['self'][2],
                 (abs(size_x - 1 - game_state['self'][3][0]), game_state['self'][3][1]))
    y['self'] = (game_state['self'][0], game_state['self'][1], game_state['self'][2],
                 (game_state['self'][3][0], abs(size_y - 1 - game_state['self'][3][1])))
    xy['self'] = (game_state['self'][0], game_state['self'][1], game_state['self'][2],
                  (abs(size_x - 1 - game_state['self'][3][0]), abs(size_y - 1 - game_state['self'][3][1])))

    for i in range(len(game_state['others'])):
        x['others'][i] = (game_state['others'][i][0], game_state['others'][i][1], game_state['others'][i][2],
                          (abs(size_x - 1 - game_state['others'][i][3][0]), game_state['others'][i][3][1]))
        y['others'][i] = (game_state['others'][i][0], game_state['others'][i][1], game_state['others'][i][2],
                          (game_state['others'][i][3][0], abs(size_y - 1 - game_state['others'][i][3][1])))
        xy['others'][i] = (game_state['others'][i][0], game_state['others'][i][1], game_state['others'][i][2],
                           (abs(size_x - 1 - game_state['others'][i][3][0]),
                            abs(size_y - 1 - game_state['others'][i][3][1])))

    return x, y, xy


def mirror_action(action: str) -> (str, str, str):
    match action:
        case 'UP':
            return 'DOWN', 'UP', 'DOWN'
        case 'RIGHT':
            return 'RIGHT', 'LEFT', 'LEFT'
        case 'DOWN':
            return 'UP', 'DOWN', 'UP'
        case 'LEFT':
            return 'LEFT', 'RIGHT', 'RIGHT'
        case _:
            return action, action, action


def mirror_feature_vector(feature_vector: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):

    x = copy.deepcopy(feature_vector)
    y = copy.deepcopy(feature_vector)
    xy = copy.deepcopy(feature_vector)

    x[2] = feature_vector[4]
    xy[2] = feature_vector[4]

    y[3] = feature_vector[5]
    xy[3] = feature_vector[5]

    x[4] = feature_vector[2]
    xy[4] = feature_vector[2]

    y[5] = feature_vector[3]
    xy[5] = feature_vector[3]

    x[8] = feature_vector[10]
    xy[8] = feature_vector[10]

    y[9] = feature_vector[11]
    xy[9] = feature_vector[11]

    x[10] = feature_vector[8]
    xy[10] = feature_vector[8]

    y[11] = feature_vector[9]
    xy[11] = feature_vector[9]

    x[12] = feature_vector[14]
    xy[12] = feature_vector[14]

    y[13] = feature_vector[15]
    xy[13] = feature_vector[15]

    x[14] = feature_vector[12]
    xy[14] = feature_vector[12]

    y[15] = feature_vector[13]
    xy[15] = feature_vector[13]

    return x, y, xy
