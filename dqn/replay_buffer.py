from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass
class DQNTrainBatch:
    batch_size: int
    state_batch: torch.Tensor
    new_state_batch: torch.Tensor
    reward_batch: torch.Tensor
    terminal_batch: torch.Tensor
    action_batch: torch.Tensor
    prev_action_batch: Optional[torch.Tensor] = None
    lengths: Optional[torch.Tensor] = None
    final_action_batch: Optional[torch.Tensor] = None


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
        action_batch = torch.tensor(self.action_mem[batch_indexes], device=device)
        return DQNTrainBatch(
            batch_size,
            state_batch,
            new_state_batch,
            reward_batch,
            terminal_batch,
            action_batch,
        )


class ReplayBufferForLSTM:
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
        self.prev_action_mem = np.zeros(self._max_buffer_size, dtype=np.int32)
        self.reward_mem = np.zeros(self._max_buffer_size, dtype=np.float32)
        self.terminal_mem = np.zeros(self._max_buffer_size, dtype=bool)
        self.episode_start_mem = np.zeros(self._max_buffer_size, dtype=np.int32)
        self.end_index = -1

    def store_transition(self, prev_action, action, state, reward, new_state, done):
        index = self.size % self._max_buffer_size
        self.state_mem[index] = state
        self.new_state_mem[index] = new_state
        self.action_mem[index] = action
        self.prev_action_mem[index] = prev_action
        self.reward_mem[index] = reward
        self.terminal_mem[index] = done
        self.size += 1

    def get_batch(self, batch_size: int, device) -> DQNTrainBatch:
        max_mem = min(self.size, self._max_buffer_size)
        batch_indexes = np.random.choice(max_mem, batch_size, replace=False)

        state_batch = []
        new_state_batch = []
        reward_batch = []
        terminal_batch = []
        action_batch = []
        prev_action_batch = []
        final_action_batch = []
        max_len = 0
        lengths = []

        for end_index in batch_indexes:
            lengths.append(abs(self.episode_start_mem[end_index] - end_index + 1))
            max_len = max(lengths[-1], max_len)

        for end_index in batch_indexes:
            indexes = _get_range_for_cyclic(
                self.episode_start_mem[end_index], end_index, self.size
            )
            rev_indexes = indexes[::-1]
            state_batch.append(_pad_seq_to_len(self.state_mem[rev_indexes], max_len))
            new_state_batch.append(
                _pad_seq_to_len(self.new_state_mem[rev_indexes], max_len)
            )
            reward_batch.append(self.reward_mem[rev_indexes[-1]])
            terminal_batch.append(self.terminal_mem[rev_indexes[-1]])
            action_batch.append(_pad_seq_to_len(self.action_mem[rev_indexes], max_len))
            prev_action_batch.append(
                _pad_seq_to_len(self.prev_action_mem[rev_indexes], max_len)
            )
            final_action_batch.append(self.action_mem[end_index])

        state_batch = torch.tensor(self.state_mem[batch_indexes], device=device)
        new_state_batch = torch.tensor(self.new_state_mem[batch_indexes], device=device)
        reward_batch = torch.tensor(self.reward_mem[batch_indexes], device=device)
        terminal_batch = torch.tensor(self.terminal_mem[batch_indexes], device=device)
        action_batch = self.action_mem[batch_indexes]

        state_batch = torch.tensor(np.stack(state_batch), device=device)
        new_state_batch = torch.tensor(np.stack(new_state_batch), device=device)
        reward_batch = torch.tensor(reward_batch, device=device)
        terminal_batch = torch.tensor(terminal_batch, device=device)
        action_batch = torch.tensor(
            np.stack(action_batch), dtype=torch.int64, device=device
        )
        prev_action_batch = torch.tensor(
            np.stack(prev_action_batch), dtype=torch.int64, device=device
        )
        lengths = torch.tensor(lengths, device=device)
        final_action_batch = torch.tensor(final_action_batch, device=device)

        return DQNTrainBatch(
            batch_size,
            state_batch,
            new_state_batch,
            reward_batch,
            terminal_batch,
            action_batch,
            prev_action_batch=prev_action_batch,
            lengths=lengths,
            final_action_batch=final_action_batch,
        )

    def episode_reset(self):
        self.end_index = self.size % self._max_buffer_size

    def episode_done(self):
        begin_index = (self.size - 1) % self._max_buffer_size
        indexes = _get_range_for_cyclic(begin_index, self.end_index, self.size)
        rev_indexes = indexes[::-1]
        self.state_mem[indexes] = self.state_mem[rev_indexes]
        self.new_state_mem[indexes] = self.new_state_mem[rev_indexes]
        self.action_mem[indexes] = self.action_mem[rev_indexes]
        self.prev_action_mem[indexes] = self.prev_action_mem[rev_indexes]
        self.reward_mem[indexes] = self.reward_mem[rev_indexes]
        self.terminal_mem[indexes] = self.terminal_mem[rev_indexes]
        self.episode_start_mem[indexes] = begin_index


def _get_range_for_cyclic(
    begin_index: int, end_index: int, array_size: int
) -> np.ndarray:
    if end_index > begin_index:
        begin_index += array_size
    indexes = np.arange(end_index, begin_index + 1) % array_size
    return indexes


def _pad_seq_to_len(seq: np.ndarray, seq_len: int) -> np.ndarray:
    pad_width = tuple([(0, seq_len - seq.shape[0])] + [(0, 0)] * (len(seq.shape) - 1))
    return np.pad(seq, pad_width)
