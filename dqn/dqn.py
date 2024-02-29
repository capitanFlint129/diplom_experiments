import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import TrainConfig


class DQN(nn.Module):
    def __init__(
        self,
        observation_size: int,
        fc_dims: int,
        n_actions: int,
    ):
        super(DQN, self).__init__()
        self.q_net = nn.Sequential(
            nn.Linear(observation_size, fc_dims),
            nn.ReLU(),
            nn.Linear(fc_dims, fc_dims),
            nn.ReLU(),
            nn.Linear(fc_dims, fc_dims),
            nn.ReLU(),
            nn.Linear(fc_dims, n_actions),
        )

    def forward(self, observation: torch.Tensor) -> torch.LongTensor:
        return self.q_net(observation)


class Agent(nn.Module):
    def __init__(
        self, observation_size: int, n_actions: int, config: TrainConfig, device
    ):
        super(Agent, self).__init__()
        self.eps_dec = config.epsilon_dec
        self.eps_end = config.epsilon_end
        self.max_mem_size = config.max_mem_size
        self.replace_target_cnt = config.replace
        self.gamma = config.gamma
        self.epsilon = config.epsilon
        self.eps_end = config.epsilon_end
        self.eps_dec = config.epsilon_dec
        self.n_actions = n_actions
        self.action_space = [i for i in range(n_actions)]
        self.max_mem_size = config.max_mem_size
        self.batch_size = config.batch_size
        self.learn_memory_threshold = config.learn_memory_threshold
        # keep track of position of first available memory
        self.mem_cntr = 0
        self.Q_eval = DQN(
            observation_size=observation_size,
            fc_dims=config.fc_dim,
            n_actions=self.n_actions,
        )
        self.Q_next = DQN(
            observation_size=observation_size,
            fc_dims=config.fc_dim,
            n_actions=self.n_actions,
        )
        self.actions_taken = []
        self.state_mem = np.zeros(
            (self.max_mem_size, observation_size), dtype=np.float32
        )
        self.new_state_mem = np.zeros(
            (self.max_mem_size, observation_size), dtype=np.float32
        )
        self.action_mem = np.zeros(self.max_mem_size, dtype=np.int32)
        self.reward_mem = np.zeros(self.max_mem_size, dtype=np.float32)
        self.terminal_mem = np.zeros(self.max_mem_size, dtype=bool)
        self.learn_step_counter = 0
        self.device = device
        self.to(self.device)

        self.optimizer = optim.Adam(self.Q_eval.parameters(), lr=config.lr)
        self.loss = nn.HuberLoss()

    def episode_reset(self):
        self.actions_taken = []

    def episode_done(self):
        pass

    def store_transition(self, action, state, reward, new_state, done):
        index = self.mem_cntr % self.max_mem_size
        self.state_mem[index] = state
        self.new_state_mem[index] = new_state
        self.action_mem[index] = action
        self.reward_mem[index] = reward
        self.terminal_mem[index] = done
        self.mem_cntr += 1

    @torch.no_grad()
    def choose_action(
        self,
        observation,
        enable_epsilon_greedy: bool = True,
        forbidden_actions: set = None,
    ):
        if forbidden_actions is None:
            forbidden_actions = set()
        if forbidden_actions is not None and len(forbidden_actions) >= len(
            self.action_space
        ):
            print("Warning: all actions are forbidden, choose 0", file=sys.stderr)
            return 0
        action = 0
        if np.random.random() <= self.epsilon and enable_epsilon_greedy:
            while action in forbidden_actions:
                action = np.random.choice(self.action_space)
        else:
            observation = observation.astype(np.float32)
            actions_q = self.Q_eval(
                torch.tensor(observation, device=self.device)[None, ...]
            )
            actions_q = actions_q.squeeze()
            action = torch.argmax(actions_q).item()
            while action in forbidden_actions:
                action = torch.argmax(actions_q).item()
                actions_q[action] = float("-inf")
        self.actions_taken.append(action)
        return action

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.Q_next.load_state_dict(self.Q_eval.state_dict())
            self.Q_next.eval()

    def learn(self):
        if self.mem_cntr < self.learn_memory_threshold:
            return
        self.optimizer.zero_grad()
        self.replace_target_network()
        max_mem = min(self.mem_cntr, self.max_mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        state_batch = torch.tensor(self.state_mem[batch], device=self.device)
        new_state_batch = torch.tensor(self.new_state_mem[batch], device=self.device)
        reward_batch = torch.tensor(self.reward_mem[batch], device=self.device)
        terminal_batch = torch.tensor(self.terminal_mem[batch], device=self.device)
        action_batch = self.action_mem[batch]
        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        with torch.no_grad():
            q_next = self.Q_next.forward(new_state_batch).max(dim=1)[0]
        q_next[terminal_batch] = 0.0
        q_target = reward_batch + self.gamma * q_next
        loss = self.loss(q_target, q_eval).to(self.device)
        loss_val = loss.item()
        loss.backward()
        self.optimizer.step()
        self.learn_step_counter += 1

        if self.epsilon > self.eps_end:
            self.epsilon -= self.eps_dec
        else:
            self.epsilon = self.eps_end
        return loss_val
