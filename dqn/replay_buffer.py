from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class DQNTrainBatch:
    state_batch: torch.Tensor
    new_state_batch: torch.Tensor
    reward_batch: torch.Tensor
    terminal_batch: torch.Tensor
    action_batch: np.ndarray


class ReplayBuffer:
    def __init__(self, buffer_size: int, observation_size: int):
        self._max_buffer_size = buffer_size
        self.size = 0

        self.state_mem = np.zeros(
            (self._max_buffer_size, observation_size), dtype=np.float32
        )
        self.new_state_mem = np.zeros(
            (self._max_buffer_size, observation_size), dtype=np.float32
        )
        self.action_mem = np.zeros(self._max_buffer_size, dtype=np.int32)
        self.reward_mem = np.zeros(self._max_buffer_size, dtype=np.float32)
        self.terminal_mem = np.zeros(self._max_buffer_size, dtype=bool)

    def store_transition(self, action, state, reward, new_state, done):
        index = self.size % self._max_buffer_size
        self.state_mem[index] = state
        self.new_state_mem[index] = new_state
        self.action_mem[index] = action
        self.reward_mem[index] = reward
        self.terminal_mem[index] = done
        self.size += 1

    def get_batch(self, batch_size: int, device) -> DQNTrainBatch:
        max_mem = min(self.size, self._max_buffer_size)
        batch_indexes = np.random.choice(max_mem, batch_size, replace=False)
        state_batch = torch.tensor(self.state_mem[batch_indexes], device=device)
        new_state_batch = torch.tensor(self.new_state_mem[batch_indexes], device=device)
        reward_batch = torch.tensor(self.reward_mem[batch_indexes], device=device)
        terminal_batch = torch.tensor(self.terminal_mem[batch_indexes], device=device)
        action_batch = self.action_mem[batch_indexes]
        return DQNTrainBatch(
            state_batch, new_state_batch, reward_batch, terminal_batch, action_batch
        )
