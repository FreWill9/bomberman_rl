import numpy as np
from random import shuffle
from collections import deque
from IPython import display
import matplotlib.pyplot as plt
import copy

DIRECTIONS = ((0, -1), (1, 0), (0, 1), (-1, 0))

def find_closest_target(game_state: dict, start: (int, int), targets: list[(int, int)]) -> list[(int, int)] | None:
    """
    Find the closest reachable target from the start position using BFS.

    Args:
        game_state: the current game state.
        start: the starting position.
        targets: the list of possible target positions.
    Returns:
        the path to the closest target excluding the starting position or None if no target is reachable.
    """
    if len(targets) == 0:
        return None

    # Use BFS to find the closest reachable coin.
    tile_queue = deque([(start[0], start[1], 0)])
    parents = {start: start}

    best = None

    while len(tile_queue) > 0:
        x, y, step = tile_queue.popleft()

        if any([x == t[0] and y == t[1] for t in targets]):
            best = (x, y, step)
            break

        if passable(x + 1, y, game_state) and (x + 1, y) not in parents:
            tile_queue.append((x + 1, y, step + 1))
            parents[(x + 1, y)] = (x, y)

        if passable(x - 1, y, game_state) and (x - 1, y) not in parents:
            tile_queue.append((x - 1, y, step + 1))
            parents[(x - 1, y)] = (x, y)

        if passable(x, y + 1, game_state) and (x, y + 1) not in parents:
            tile_queue.append((x, y + 1, step + 1))
            parents[(x, y + 1)] = (x, y)

        if passable(x, y - 1, game_state) and (x, y - 1) not in parents:
            tile_queue.append((x, y - 1, step + 1))
            parents[(x, y - 1)] = (x, y)

    if best is None:
        return None

    path = []
    current = best[0:1]
    while current != start:
        path.append(current)
        current = parents[current]

    path.reverse()
    return path


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
    if coord_target == (x, y - 1):
        return 'UP', 1.0, 0.0, 0.0, 0.0
    if coord_target == (x + 1, y):
        return 'RIGHT', 0.0, 1.0, 0.0, 0.0
    if coord_target == (x, y + 1):
        return 'DOWN', 0.0, 0.0, 1.0, 0.0
    if coord_target == (x - 1, y):
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
    plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores) - 1, mean_scores[-1], str(mean_scores[-1]))
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
            return 'UP', 'DOWN', 'DOWN'
        case 'RIGHT':
            return 'LEFT', 'RIGHT', 'LEFT'
        case 'DOWN':
            return 'DOWN', 'UP', 'UP'
        case 'LEFT':
            return 'RIGHT', 'LEFT', 'RIGHT'
        case _:
            return action, action, action


def mirror_feature_vector(feature_vector: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    (in_danger, bomb_avail, up, right, down, left, touching_crate, first_step, bomb_for_trap,
     shortest_way_coin_up, shortest_way_coin_right,
     shortest_way_coin_down, shortest_way_coin_left,
     shortest_way_safety_up, shortest_way_safety_right,
     shortest_way_safety_down, shortest_way_safety_left,
     shortest_way_trap_up, shortest_way_trap_right,
     shortest_way_trap_down, shortest_way_trap_left,
     explosion_score_up, explosion_score_right,
     explosion_score_down, explosion_score_left, explosion_score_stay) = tuple(feature_vector)

    x = np.array([in_danger, bomb_avail, up, left, down, right, touching_crate, first_step, bomb_for_trap,
                  shortest_way_coin_up, shortest_way_coin_left,
                  shortest_way_coin_down, shortest_way_coin_right,
                  shortest_way_safety_up, shortest_way_safety_left,
                  shortest_way_safety_down, shortest_way_safety_right,
                  shortest_way_trap_up, shortest_way_trap_left,
                  shortest_way_trap_down, shortest_way_trap_right,
                  explosion_score_up, explosion_score_left,
                  explosion_score_down, explosion_score_right, explosion_score_stay])
    y = np.array([in_danger, bomb_avail, down, right, up, left, touching_crate, first_step, bomb_for_trap,
                  shortest_way_coin_down, shortest_way_coin_right,
                  shortest_way_coin_up, shortest_way_coin_left,
                  shortest_way_safety_down, shortest_way_safety_right,
                  shortest_way_safety_up, shortest_way_safety_left,
                  shortest_way_trap_down, shortest_way_trap_right,
                  shortest_way_trap_up, shortest_way_trap_left,
                  explosion_score_down, explosion_score_right,
                  explosion_score_up, explosion_score_left, explosion_score_stay])
    xy = np.array([in_danger, bomb_avail, down, left, up, right, touching_crate, first_step, bomb_for_trap,
                   shortest_way_coin_down, shortest_way_coin_left,
                   shortest_way_coin_up, shortest_way_coin_right,
                   shortest_way_safety_down, shortest_way_safety_left,
                   shortest_way_safety_up, shortest_way_safety_right,
                   shortest_way_trap_down, shortest_way_trap_left,
                   shortest_way_trap_up, shortest_way_trap_right,
                   explosion_score_down, explosion_score_left,
                   explosion_score_up, explosion_score_right, explosion_score_stay])

    return x, y, xy


def passable(x, y, game_state):
    return (in_field(x, y, game_state) and game_state['field'][x, y] == 0
            and (x, y) not in [xy for xy, t in game_state['bombs']]
            and (x, y) not in [xy for (n, s, b, xy) in game_state['others']])


def closest_target_dist(game_state: dict, coord: (int, int), targets: list[(int, int)] = None) -> int:
    """
    Calculate the distance to the closest target from the coordinate using BFS.
    """
    if targets is None:
        targets = game_state['coins']

    if len(targets) == 0:
        return 10_000

    # Use BFS to find the closest reachable coin.
    tile_queue = deque([(coord[0], coord[1], 0)])
    visited = np.zeros(game_state['field'].shape)
    visited[coord[0], coord[1]] = 1
    while len(tile_queue) > 0:
        x, y, step = tile_queue.popleft()
        if any([x == t[0] and y == t[1] for t in targets]):
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

    return 10_000


def best_explosion_score(game_state: dict, bomb_map, coord: (int, int), direction: (int, int), max_step: int) -> int:
    """
    Get the highest explosion score for any tile reachable in max_step steps in the specified direction.
    """
    coins = game_state['coins']
    if len(coins) == 0:
        return 0

    # Use BFS
    tile_queue = deque([(coord[0] + direction[0], coord[1] + direction[1], 1)])
    visited = np.zeros(game_state['field'].shape)
    visited[coord[0], coord[1]] = 1
    visited[coord[0] + direction[0], coord[1] + direction[1]] = 1
    best_score = 0
    while len(tile_queue) > 0:
        x, y, step = tile_queue.popleft()

        best_score = max(best_score, explosion_score(game_state, bomb_map, x, y))

        if step >= max_step:
            continue

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

    return best_score


def explosion_score(game_state: dict, bomb_map, x: int, y: int) -> float:
    crate_score = 0
    for i in range(1, 4):
        if in_field(x + i, y, game_state) and game_state['field'][x + i, y] != -1:
            if game_state['field'][x + i, y] == 1 and bomb_map[x + i, y] == 100 and \
                    game_state['explosion_map'][x + i, y] == 0:
                crate_score += 1
        else:
            break
    for i in range(1, 4):
        if in_field(x - i, y, game_state) and game_state['field'][x - i, y] != -1:
            if game_state['field'][x - i, y] == 1 and bomb_map[x - i, y] == 100 and \
                    game_state['explosion_map'][x - i, y] == 0:
                crate_score += 1
        else:
            break
    for i in range(1, 4):
        if in_field(x, y + i, game_state) and game_state['field'][x, y + i] != -1:
            if game_state['field'][x, y + i] == 1 and bomb_map[x, y + i] == 100 and \
                    game_state['explosion_map'][x, y + i] == 0:
                crate_score += 1
        else:
            break
    for i in range(1, 4):
        if in_field(x, y - i, game_state) and game_state['field'][x, y - i] != -1:
            if game_state['field'][x, y - i] == 1 and bomb_map[x, y - i] == 100 and \
                    game_state['explosion_map'][x, y - i] == 0:
                crate_score += 1
        else:
            break

    return crate_score / 10


def safe_tile_reachable(x, y, escape_space, safe_tiles) -> bool:
    dir_safety = look_for_targets(escape_space, (x, y), safe_tiles)
    if dir_safety is None or (x, y) == dir_safety:
        return False
    else:
        return True


def find_traps(game_state: dict, empty_tiles, others: list[(int, int)]) -> (list, list):
    arena = game_state['field']
    explosions = game_state['explosion_map']
    self_coord = game_state['self'][3]

    trap_tiles = set([])
    bomb_for_trap_tiles = set([])

    if closest_target_dist(game_state, self_coord, others) < 8:
        for x, y in empty_tiles:
            pot_game_state = copy.deepcopy(game_state)
            pot_game_state['bombs'].append(((x, y), 5))
            pot_bomb_map = build_bomb_map(pot_game_state)
            pot_bomb_xys = [xy for (xy, t) in pot_game_state['bombs']]
            pot_escape_tiles = [tile for tile in empty_tiles if explosions[tile[0], tile[1]] == 0 and \
                                tile not in pot_bomb_xys]
            pot_escape_space = np.zeros((arena.shape[0], arena.shape[1]), dtype=bool)
            for tile in pot_escape_tiles:
                pot_escape_space[tile[0], tile[1]] = True
            pot_safe_tiles = [tile for tile in empty_tiles if pot_bomb_map[tile[0], tile[1]] == 100 and \
                              explosions[tile[0], tile[1]] == 0]
            for i, j in empty_tiles:
                if pot_bomb_map[i, j] < 100 and not safe_tile_reachable(i, j, pot_escape_space, pot_safe_tiles):
                    trap_tiles.add((i, j))
                    if (i, j) in others:
                        bomb_for_trap_tiles.add((x, y))

    return list(trap_tiles), list(bomb_for_trap_tiles)
