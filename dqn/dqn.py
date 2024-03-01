import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import TrainConfig
from dqn.q_value import DQN
from dqn.replay_buffer import ReplayBuffer


class Agent:
    def __init__(
        self, observation_size: int, n_actions: int, config: TrainConfig, device
    ):
        self._config = config
        self._replay_buffer = ReplayBuffer(
            buffer_size=config.max_mem_size,
            observation_size=config.observation_size,
        )
        self._epsilon = config.epsilon
        self._n_actions = n_actions
        self.Q_eval = DQN(
            observation_size=observation_size,
            fc_dims=config.fc_dim,
            n_actions=self._n_actions,
        ).to(device)
        self.Q_next = DQN(
            observation_size=observation_size,
            fc_dims=config.fc_dim,
            n_actions=self._n_actions,
        ).to(device)
        self._actions_taken = []
        self._learn_step_counter = 0
        self._device = device

        self._optimizer = optim.Adam(self.Q_eval.parameters(), lr=config.lr)
        self._loss = nn.HuberLoss()

    def episode_reset(self):
        self._actions_taken = []

    def episode_done(self):
        pass

    @torch.no_grad()
    def choose_action(
        self,
        observation,
        enable_epsilon_greedy: bool = True,
        forbidden_actions: set = None,
    ):
        if forbidden_actions is None:
            forbidden_actions = set()
        if forbidden_actions is not None and len(forbidden_actions) >= self._n_actions:
            print("Warning: all actions are forbidden, choose 0", file=sys.stderr)
            return 0
        action = 0
        if np.random.random() <= self._epsilon and enable_epsilon_greedy:
            while action in forbidden_actions:
                action = np.random.choice(self._n_actions)
        else:
            observation = observation.astype(np.float32)
            actions_q = self.Q_eval(
                torch.tensor(observation, device=self._device)[None, ...]
            )
            actions_q = actions_q.squeeze()
            action = torch.argmax(actions_q).item()
            while action in forbidden_actions:
                action = torch.argmax(actions_q).item()
                actions_q[action] = float("-inf")
        self._actions_taken.append(action)
        return action

    def learn(self):
        if self._replay_buffer.size < self._config.learn_memory_threshold:
            return
        self._optimizer.zero_grad()
        self._replace_target_network()
        dqn_batch = self._replay_buffer.get_batch(self._config.batch_size, self._device)
        q_eval = self.Q_eval.forward(dqn_batch.state_batch)[:, dqn_batch.action_batch]
        with torch.no_grad():
            q_next = self.Q_next.forward(dqn_batch.new_state_batch).max(dim=1)[0]
        q_next[dqn_batch.terminal_batch] = 0.0
        q_target = dqn_batch.reward_batch + self._config.gamma * q_next
        loss = self._loss(q_target, q_eval).to(self._device)
        loss_val = loss.item()
        loss.backward()
        self._optimizer.step()
        self._learn_step_counter += 1
        if self._epsilon > self._config.epsilon_end:
            self._epsilon -= self._config.epsilon_dec
        else:
            self._epsilon = self._config.epsilon_end
        return loss_val

    def store_transition(self, action, observation, reward, new_observation, done):
        self._replay_buffer.store_transition(
            action, observation, reward, new_observation, done
        )

    def eval(self):
        self.Q_eval.eval()
        self.Q_next.eval()

    def train(self):
        self.Q_eval.train()
        self.Q_next.train()

    def _replace_target_network(self):
        if self._learn_step_counter % self._config.replace == 0:
            self.Q_next.load_state_dict(self.Q_eval.state_dict())
            self.Q_next.eval()
