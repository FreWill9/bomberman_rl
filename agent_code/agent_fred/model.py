from collections import deque
import random

import torch
from torch import optim, nn, tensor
from torch.nn import functional as F


class QNet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size
        self.linear1 = nn.Linear(input_size, hidden_size1)
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)
        self.linear3 = nn.Linear(hidden_size2, hidden_size2)
        self.linear4 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear4(x)
        return x

    def clone(self):
        net = QNet(self.input_size, self.hidden_size1, self.hidden_size2, self.output_size)
        net.load_state_dict(self.state_dict())
        return net

class QNet2(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size
        self.linear1 = nn.Linear(input_size, hidden_size1)
        self.linear2 = nn.Linear(hidden_size1, hidden_size1)
        self.linear3 = nn.Linear(hidden_size1, hidden_size2)
        self.linear4 = nn.Linear(hidden_size2, hidden_size2)
        self.linear5 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        # x = F.relu(self.linear4(x))
        x = self.linear5(x)
        return x

    def clone(self):
        net = QNet2(self.input_size, self.hidden_size1, self.hidden_size2, self.output_size)
        net.load_state_dict(self.state_dict())
        return net


class DQN:
    def __init__(self, model: QNet, lr, gamma, device):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.SmoothL1Loss()
        self.device = device

    def train(self, state, action, reward, next_state, done):
        state = tensor(state, dtype=torch.float).to(self.device)
        action = tensor(action, dtype=torch.float).to(self.device)
        reward = tensor(reward, dtype=torch.float).to(self.device)
        next_state = tensor(next_state, dtype=torch.float).to(self.device)

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


class DQN2:
    def __init__(self, model: QNet, lr, gamma, learning_rate, batch_size, max_memory, device):
        self.lr = lr
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.model = model
        self.target_model = model.clone().to(device)
        self.batch_size = batch_size
        self.memory = deque(maxlen=max_memory)
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.SmoothL1Loss()
        self.step = 0
        self.device = device

    def train(self, state, action, reward, next_state, done):
        self.step += 1
        self.memory.append((state, action, reward, next_state, done))

        if len(self.memory) < self.batch_size:
            return

        if self.step % 16 == 0:
            self._update()

        if self.step % 400 == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def _update(self):
        mini_sample = random.sample(self.memory, self.batch_size)

        states, actions, rewards, next_states, dones = zip(*mini_sample)

        states = tensor(states, dtype=torch.float).to(self.device)
        actions = tensor(actions, dtype=torch.float).to(self.device)
        rewards = tensor(rewards, dtype=torch.float).to(self.device)
        next_states = tensor(next_states, dtype=torch.float).to(self.device)

        pred = self.model(states)
        target_model_pred = self.target_model(next_states)

        target = pred.clone()

        for i in range(len(dones)):
            if not dones[i]:
                q_new = rewards[i] + self.gamma * torch.max(target_model_pred[i])
            else:
                q_new = rewards[i]

            action_idx = int(actions[i].item())
            target[i][action_idx] = q_new

        self.optimizer.zero_grad()
        loss = self.criterion(pred, target)
        loss.backward()

        self.optimizer.step()
