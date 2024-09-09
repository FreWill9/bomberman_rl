from collections import namedtuple, deque
import pickle
from typing import List
import events as e
import random
import matplotlib.pyplot as plt
import torch

from .callbacks import state_to_features, Linear_QNet, QTrainer
from .helpers import transpose_action, encode_action, plot
from .custom_events import *

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu")

# This is only an example!
Memory = namedtuple('Memory',
                    ('state', 'action', 'reward', 'next_state', 'done'))

# Hyperparameters -- DO modify
MAX_MEMORY = 10_000
BATCH_SIZE = 128
LR = 0.001

plot_scores = []
plot_mean_scores = []
total_score = 0


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks, and you can set arbitrary values.
    """
    # Example: Set up an array that will remember transition tuples
    # (s, a, r, s_new)
    # self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)

    self.plot_scores = []
    self.plot_mean_scores = []
    self.total_score = 0
    self.n_games = 0
    self.epsilon = 0
    self.gamma = 0.95
    self.memory = deque(maxlen=MAX_MEMORY)
    self.model = Linear_QNet(9, 64, 64, 6).to(device)
    self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks, and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    state_old = old_game_state
    action = encode_action(self_action)
    state_new = new_game_state

    # Custom events to hand out rewards:
    # Agent did not wait
    if 'WAIT' not in events:
        events.append(NOT_WAITED)

    # If agent has been in the same location three times recently, it's a loop
    if self.coordinate_history.count((state_new['self'][3][0], state_new['self'][3][1])) > 2:
        events.append(LOOP)
    else:
        events.append(NO_LOOP)

    # Taking the shortest path to the next coin
    shortest_way_coin = self.shortest_way_coin
    # Transpose shortest_way_coin to match gui
    shortest_way_coin = transpose_action(shortest_way_coin)

    if shortest_way_coin == "No More Coins" or self.shortest_way_safety != 'Not In Danger':
        pass
    elif self_action == shortest_way_coin:
        events.append(SHORTEST_WAY_COIN)
    else:
        events.append(NOT_SHORTEST_WAY_COIN)

    # Taking the shortest path to the next crate
    shortest_way_crate = self.shortest_way_crate
    # Transpose shortest_way_crate to match gui
    shortest_way_crate = transpose_action(shortest_way_crate)

    if shortest_way_crate == "No More Crates" or self.shortest_way_safety != 'Not In Danger':
        pass
    elif self_action == shortest_way_crate:
        events.append(SHORTEST_WAY_CRATE)
    else:
        events.append(NOT_SHORTEST_WAY_CRATE)

    # Taking the shortest path out of danger
    shortest_way_safety = self.shortest_way_safety
    # Transpose shortest_way_safety to match gui
    shortest_way_safety = transpose_action(shortest_way_safety)

    if shortest_way_safety == "Not In Danger":
        pass
    elif self_action == shortest_way_safety:
        events.append(SHORTEST_WAY_SAFETY)
    else:
        events.append(NOT_SHORTEST_WAY_SAFETY)

    rewards = reward_from_events(self, events, state_new['self'][1])

    # state_to_features is defined in callbacks.py
    # remember
    self.memory.append(Memory(state_to_features(self, state_old, self.coordinate_history), action, rewards, state_to_features(self, state_new, self.coordinate_history), False))

    # train short memory
    self.trainer.train_step(state_to_features(self, state_old, self.coordinate_history), action, rewards, state_to_features(self, state_new, self.coordinate_history), False)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    state_old = last_game_state
    action = encode_action(last_action)

    # Add custom events here
    score = state_old['self'][1]
    if score < 40:
        events.append(LOW_SCORING_GAME)
    if score > 45:
        events.append(HIGH_SCORING_GAME)
    if score == 49:
        events.append(PERFECT_COIN_HEAVEN)

    rewards = reward_from_events(self, events, state_old['self'][1])

    # remember Todo: last state passed twice as parameter because easier with train step... is this a problem?
    self.memory.append(
        Memory(state_to_features(self, state_old, self.coordinate_history), action, rewards, state_to_features(self, state_old, self.coordinate_history), True))

    # train long memory
    if len(self.memory) > BATCH_SIZE:
        mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
    else:
        mini_sample = self.memory

    states, actions, rewards, next_states, dones = zip(*mini_sample)
    self.trainer.train_step(states, actions, rewards, next_states, dones)
    # for state, action, reward, next_state, done in mini_sample:
    # self.trainer.train_step(state, action, reward, next_state, done)

    self.n_games += 1

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)

    # plot results
    self.plot_scores.append(score)
    self.total_score += score
    mean_score = self.total_score / self.n_games
    self.plot_mean_scores.append(mean_score)
    plt.ion()
    plot(self.plot_scores, self.plot_mean_scores)


def reward_from_events(self, events: List[str], score) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent gets to en/discourage
    certain behavior.
    """
    # Stage 1 finished
    game_rewards_stage_1 = {
        e.COIN_COLLECTED: +4,
        e.KILLED_SELF: -2,
        e.INVALID_ACTION: -3,
        e.WAITED: -2,
        e.SURVIVED_ROUND: +10,
        e.BOMB_DROPPED: -1,
        LOOP: -3,
        NO_LOOP: +1,
        HIGH_SCORING_GAME: +10,
        PERFECT_COIN_HEAVEN: +20,
        LOW_SCORING_GAME: -0,
        SHORTEST_WAY_COIN: +5,
        NOT_SHORTEST_WAY_COIN: -3,
        SHORTEST_WAY_SAFETY: +7,
        NOT_SHORTEST_WAY_SAFETY: -6
    }

    # Stage 2
    game_rewards_stage_2 = {
        e.COIN_COLLECTED: +1,
        e.KILLED_SELF: -1,
        e.INVALID_ACTION: -1,
        e.WAITED: 0,
        e.SURVIVED_ROUND: +20,
        e.BOMB_DROPPED: -1,
        e.CRATE_DESTROYED: +10,
        e.COIN_FOUND: +30,
        e.BOMB_EXPLODED: +0,
        LOOP: -3,
        NO_LOOP: +1,
        HIGH_SCORING_GAME: +20,
        SHORTEST_WAY_COIN: +2,
        SHORTEST_WAY_CRATE: +2,
        NOT_SHORTEST_WAY_COIN: -1,
        NOT_SHORTEST_WAY_CRATE: -1
    }

    reward_sum = self.steps / 40
    for event in events:
        if event in game_rewards_stage_1:
            reward_sum += game_rewards_stage_1[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
