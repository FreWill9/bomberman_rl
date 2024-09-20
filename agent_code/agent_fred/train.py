from collections import namedtuple, deque
import pickle
from typing import List
import events as e
import random
import matplotlib.pyplot as plt
import torch

from .callbacks import state_to_features, QTrainer
from .helpers import encode_action, plot, mirror_action, mirror_feature_vector
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
MAX_MEMORY = 40_000
BATCH_SIZE = 64
LR = 0.001

plot_maxlen = 100
plt.style.use('Solarize_Light2')


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks, and you can set arbitrary values.
    """
    # Example: Set up an array that will remember transition tuples
    # (s, a, r, s_new)
    # self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)

    self.recent_scores = deque(maxlen=plot_maxlen)
    self.plot_scores = []
    self.plot_mean_scores = []
    self.total_score = 0
    self.epsilon = 0
    self.gamma = 0.95
    self.memory = deque(maxlen=MAX_MEMORY)
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

    # Custom events to hand out rewards:
    # Agent did not wait
    if 'WAIT' not in events:
        events.append(NOT_WAITED)

    # If agent has been in the same location four times recently, it's a loop
    if self.coordinate_history.count((new_game_state['self'][3][0], new_game_state['self'][3][1])) >= 4:
        events.append(LOOP)
    else:
        events.append(NO_LOOP)

    # Transpose shortest_ways to match gui
    shortest_way_coin = self.shortest_way_coin
    shortest_way_crate = self.shortest_way_crate
    shortest_way_safety = self.shortest_way_safety
    shortest_way_trap = self.shortest_way_trap

    # Taking the shortest path to the next coin
    if shortest_way_coin == "None" or self.shortest_way_safety != 'None':
        pass
    elif self_action == shortest_way_coin:
        events.append(SHORTEST_WAY_COIN)
    else:
        events.append(NOT_SHORTEST_WAY_COIN)

    # Taking the shortest path to the next crate
    if shortest_way_crate == "None" or self.shortest_way_safety != 'None' or self.shortest_way_crate == "BOMB":
        pass
    elif self_action == shortest_way_crate:
        events.append(SHORTEST_WAY_CRATE)
    else:
        events.append(NOT_SHORTEST_WAY_CRATE)

    # Taking the shortest path out of danger
    if shortest_way_safety == "None":
        pass
    elif self_action == shortest_way_safety:
        events.append(SHORTEST_WAY_SAFETY)
    else:
        events.append(NOT_SHORTEST_WAY_SAFETY)

    # Taking the shortest path to trap an opp
    if shortest_way_trap == "None" or self.shortest_way_safety != 'None':
        pass
    elif self_action == shortest_way_trap:
        events.append(SHORTEST_WAY_TRAP)
    else:
        events.append(NOT_SHORTEST_WAY_TRAP)

    # Bomb on spawn point has high prob of no escape
    if len(self.coordinate_history) == 1 and self_action == 'BOMB':
        events.append(STEP_ONE_BOMB)
    elif len(self.coordinate_history) == 1 and self_action != 'WAIT':
        events.append(NOT_STEP_ONE_BOMB)

    # Good Bombs destroy crates or put opps in danger
    if old_game_state['self'][2]:
        pass
    elif self_action == 'BOMB' and self.shortest_way_crate == 'BOMB':
        events.append(GOOD_BOMB)
    elif self_action == 'BOMB':
        events.append(BAD_BOMB)

    # Reward for trapping opp
    if self.bomb_for_trap == 1 and self_action == 'BOMB':
        events.append(TRAP)
    elif self.bomb_for_trap == 1 and self_action != 'BOMB':
        events.append(MISSED_TRAP)

    reward = reward_from_events(self, events, new_game_state['self'][1])

    # augment the dataset
    # state to features is non-deterministic, calling multiple times can cause problems
    state_old_features = self.features
    state_new_features = state_to_features(self, new_game_state)
    # mirror feature vectors and actions on x, y and both axes:
    x_old_features, y_old_features, xy_old_features = mirror_feature_vector(state_old_features)
    x_new_features, y_new_features, xy_new_features = mirror_feature_vector(state_new_features)
    x_act, y_act, xy_act = mirror_action(self_action)

    # encode_action
    action_enc = encode_action(self_action)
    x_act_enc = encode_action(x_act)
    y_act_enc = encode_action(y_act)
    xy_act_enc = encode_action(xy_act)

    # remember
    self.memory.append(Memory(state_old_features, action_enc, reward, state_new_features, False))
    self.memory.append(Memory(x_old_features, x_act_enc, reward, x_new_features, False))
    self.memory.append(Memory(y_old_features, y_act_enc, reward, y_new_features, False))
    self.memory.append(Memory(xy_old_features, xy_act_enc, reward, xy_new_features, False))

    # train short term memory
    # self.trainer.train_step(state_old_features, action_enc, reward, state_new_features, False)
    # self.trainer.train_step(x_old_features, x_act_enc, reward, x_new_features, False)
    # self.trainer.train_step(y_old_features, y_act_enc, reward, y_new_features, False)
    # self.trainer.train_step(xy_old_features, xy_act_enc, reward, xy_new_features, False)

    if self.step % 4 == 0:
        # train long term memory
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

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

    # Add custom events here
    score = last_game_state['self'][1]
    if score < 40:
        events.append(LOW_SCORING_GAME)
    if score > 45:
        events.append(HIGH_SCORING_GAME)
    if score >= 49:
        events.append(PERFECT_COIN_HEAVEN)

    match self.placement:
        case 1:
            events.append("FIRST")
        case 2:
            events.append("SECOND")
        case 3:
            events.append("THIRD")
        case 4:
            events.append("FOURTH")

    reward = reward_from_events(self, events, last_game_state['self'][1])

    # augment the dataset
    # state to features not needed
    last_state_features = self.features
    # mirror game-states and actions on x, y and both axes:
    x_last_features, y_last_features, xy_last_features = mirror_feature_vector(last_state_features)
    x_act, y_act, xy_act = mirror_action(last_action)

    # encode actions
    last_action_enc = encode_action(last_action)
    x_act_enc = encode_action(x_act)
    y_act_enc = encode_action(y_act)
    xy_act_enc = encode_action(xy_act)

    # remember
    self.memory.append(Memory(last_state_features, last_action_enc, reward, last_state_features, True))
    self.memory.append(Memory(x_last_features, x_act_enc, reward, x_last_features, True))
    self.memory.append(Memory(y_last_features, y_act_enc, reward, y_last_features, True))
    self.memory.append(Memory(xy_last_features, xy_act_enc, reward, xy_last_features, True))

    # train short term memory
    self.trainer.train_step(last_state_features, last_action_enc, reward, last_state_features, True)
    # self.trainer.train_step(x_last_features, x_act_enc, reward, x_last_features, True)
    # self.trainer.train_step(y_last_features, y_act_enc, reward, y_last_features, True)
    # self.trainer.train_step(xy_last_features, xy_act_enc, reward, xy_last_features, True)

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)

    # plot results
    self.plot_scores.append(score)
    self.recent_scores.append(score)
    recent_mean_scores = sum(self.recent_scores) / len(self.recent_scores)
    self.plot_mean_scores.append(recent_mean_scores)
    plt.ion()
    plot(self.plot_scores, self.plot_mean_scores)


def reward_from_events(self, events: List[str], score) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent gets to en/discourage
    certain behavior.
    """
    # Stage 1
    game_rewards_stage_1 = {
        e.COIN_COLLECTED: +1,
        e.KILLED_SELF: -1,
        e.INVALID_ACTION: -1,
        e.WAITED: -1,
        e.BOMB_DROPPED: -1,
        SHORTEST_WAY_COIN: +1,
        NOT_SHORTEST_WAY_COIN: -1,
        SHORTEST_WAY_SAFETY: +1,
        NOT_SHORTEST_WAY_SAFETY: -1
    }

    # Stage 1.5
    # Remember to activate forced bomb drop in act()!
    game_rewards_stage_1_5 = {
        e.COIN_COLLECTED: +1,
        e.KILLED_SELF: -1,
        e.INVALID_ACTION: -1,
        LOOP: -1,
        NO_LOOP: +1,
        SHORTEST_WAY_COIN: +1,
        NOT_SHORTEST_WAY_COIN: -1,
        SHORTEST_WAY_SAFETY: +1,
        NOT_SHORTEST_WAY_SAFETY: -1
    }

    # Stage 2
    game_rewards_stage_2 = {
        e.COIN_COLLECTED: +1,
        e.KILLED_SELF: -3,
        e.INVALID_ACTION: -1,
        e.WAITED: -0,
        e.SURVIVED_ROUND: +0,
        e.BOMB_DROPPED: +0,
        e.CRATE_DESTROYED: +1,
        e.COIN_FOUND: +1,
        e.BOMB_EXPLODED: +0,
        STEP_ONE_BOMB: -0,
        NOT_STEP_ONE_BOMB: +0,
        GOOD_BOMB: +1,
        BAD_BOMB: -1,
        LOOP: 0,
        NO_LOOP: +0,
        SHORTEST_WAY_COIN: +1,
        NOT_SHORTEST_WAY_COIN: -1,
        SHORTEST_WAY_CRATE: +2,
        NOT_SHORTEST_WAY_CRATE: -1,
        SHORTEST_WAY_SAFETY: +1,
        NOT_SHORTEST_WAY_SAFETY: -1
    }

    # Stage 3
    game_rewards_stage_3 = {
        e.COIN_COLLECTED: +1,
        e.KILLED_SELF: -1,
        e.GOT_KILLED: -1,
        e.KILLED_OPPONENT: +3,
        e.OPPONENT_ELIMINATED: +2,
        e.INVALID_ACTION: -1,
        e.SURVIVED_ROUND: +1,
        e.CRATE_DESTROYED: +1,
        e.COIN_FOUND: +1,
        GOOD_BOMB: +1,
        BAD_BOMB: -1,
        TRAP: +10,
        MISSED_TRAP: -10,
        SHORTEST_WAY_COIN: +1,
        NOT_SHORTEST_WAY_COIN: 0,
        SHORTEST_WAY_SAFETY: +1,
        NOT_SHORTEST_WAY_SAFETY: -1,
        SHORTEST_WAY_TRAP: +2,
        NOT_SHORTEST_WAY_TRAP: -1,
    }

    reward_sum = 0
    for event in events:
        if event in game_rewards_stage_2:
            reward_sum += game_rewards_stage_2[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
