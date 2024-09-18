from collections import namedtuple, deque

import pickle
from typing import List

import events as e
from .callbacks import state_to_features, Linear_QNet, QTrainer, coin_dist_sum, coin_score

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import random

import matplotlib.pyplot as plt
from IPython import display

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu")

# This is only an example!
Memory = namedtuple('Memory',
                    ('state', 'action', 'reward', 'next_state', 'done'))

# Hyperparameters -- DO modify
# TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
# RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...
MAX_MEMORY = 10_000
BATCH_SIZE = 128
LR = 0.001

plot_scores = []
plot_mean_scores = []
total_score = 0

# Events
NOT_WAITED = "NOT_WAITED"
NO_COIN_FOR_X_MOVES = "NO_COIN_FOR_X_MOVES"
LOOP = "LOOP"
NO_LOOP = "NO_LOOP"
NO_LOOP_SCORE = "NO_LOOP_SCORE"
HIGH_SCORING_GAME = "HIGH_SCORING_GAME"
PERFECT_COIN_HEAVEN = "PERFECT_COIN_HEAVEN"
LOW_SCORING_GAME = "LOW_SCORING_GAME"
MOVED_TO_COIN = "MOVED_TO_COIN"
MOVED_AWAY_FROM_COIN = "MOVED_AWAY_FROM_COIN"
MOVED_BACK = "MOVED_BACK"

STAYED_SAFE = "STAYED_SAFE"
DOOMED_SELF = "DOOMED_SELF"

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks, and you can set arbitrary values.
    """
    # Example: Set up an array that will note transition tuples
    # (s, a, r, s')
    # self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)

    self.plot_scores = []
    self.plot_mean_scores = []
    self.total_score = 0
    self.n_games = 0
    self.epsilon = 0
    self.gamma = 0.95
    self.memory = deque(maxlen=MAX_MEMORY)
    self.model = Linear_QNet(4, 64, 64, 6).to(device)
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

    # state_to_features is defined in callbacks.py
    # remember
    features_old = state_to_features(self, state_old)
    features_new = state_to_features(self, state_new)

    # Idea: Add your own events to hand out rewards
    if 'WAIT' not in events:
        events.append(NOT_WAITED)

    # If tile value increases, agent is moving towards coin
    if len(state_old['coins']) > 0:
        if (coin_score(state_old, state_old['self'][3][0], state_old['self'][3][1])
                < coin_score(state_new, state_new['self'][3][0], state_new['self'][3][1])):
            events.append(MOVED_TO_COIN)
        else:
            events.append(MOVED_AWAY_FROM_COIN)

    # Moved safely
    if 'MOVED_UP' in events:
        if self.safety_up >= 1.0:
            events.append(STAYED_SAFE)
        elif self.safety_up == -1.0:
            events.append(DOOMED_SELF)
    elif 'MOVED_DOWN' in events:
        if self.safety_down >= 1.0:
            events.append(STAYED_SAFE)
        elif self.safety_down == -1.0:
            events.append(DOOMED_SELF)
    elif 'MOVED_LEFT' in events:
        if self.safety_left >= 1.0:
            events.append(STAYED_SAFE)
        elif self.safety_left == -1.0:
            events.append(DOOMED_SELF)
    elif 'MOVED_RIGHT' in events:
        if self.safety_right >= 1.0:
            events.append(STAYED_SAFE)
        elif self.safety_right == -1.0:
            events.append(DOOMED_SELF)
    else:
        if self.safety_stay >= 1.0:
            events.append(STAYED_SAFE)
        elif self.safety_stay == -1.0:
            events.append(DOOMED_SELF)

    rewards = reward_from_events(self, events, state_new['self'][1])

    self.memory.append(Memory(features_old, action, rewards, features_new, False))

    # train short memory
    self.trainer.train_step(features_old, action, rewards, features_new, False)


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
    rewards = reward_from_events(self, events, state_old['self'][1])

    score = state_old['self'][1]
    coins_left = len(last_game_state['coins'])
    if coins_left > 10:
        events.append(LOW_SCORING_GAME)
    elif coins_left > 0:
        events.append(HIGH_SCORING_GAME)
    else:
        events.append(PERFECT_COIN_HEAVEN)

    # same features are passed twice since new features are not available / ignored for last step
    features = state_to_features(self, state_old)
    self.memory.append(Memory(features, action, rewards, features, True))

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
    game_rewards = {
        e.COIN_COLLECTED: +10,
        e.MOVED_UP: +0,
        e.MOVED_RIGHT: +0,
        e.MOVED_DOWN: +0,
        e.MOVED_LEFT: +0,
        e.KILLED_SELF: -1000,
        e.INVALID_ACTION: -5,
        e.WAITED: -1,
        e.SURVIVED_ROUND: +0,
        e.GOT_KILLED: -100,
        e.BOMB_DROPPED: -5,
        NOT_WAITED: +0,
        # LOOP: -2,
        # NO_LOOP: +2,
        HIGH_SCORING_GAME: +50,
        PERFECT_COIN_HEAVEN: + 500,
        LOW_SCORING_GAME: +0,
        # MOVED_TO_COIN: +1,
        # MOVED_AWAY_FROM_COIN: -1,
        MOVED_BACK: -1,
        # e.KILLED_OPPONENT: -5
        # STAYED_SAFE: +3,
        DOOMED_SELF: -50
    }
    reward_sum = score // 10
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum


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
