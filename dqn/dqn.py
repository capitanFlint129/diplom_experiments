import sys
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config.config import TrainConfig
from dqn.q_value import DQN, DQNLSTM, DuelingDQN
from dqn.replay_buffer import DQNTrainBatch, ReplayBuffer, ReplayBufferForLSTM


class DQNAgent(ABC):
    @abstractmethod
    def get_epsilon(self) -> float:
        pass

    @abstractmethod
    def get_policy_net_state_dict(self) -> dict:
        pass

    @abstractmethod
    def episode_reset(self) -> None:
        pass

    @abstractmethod
    def episode_done(self) -> None:
        pass

    @abstractmethod
    def choose_action(
        self,
        observation: np.ndarray,
        enable_epsilon_greedy: bool,
        forbidden_actions: set[int],
        eval_mode: bool,
        hack: bool = False,
    ) -> tuple[int, float]:
        pass

    @abstractmethod
    def learn(self) -> Optional[float]:
        pass

    @abstractmethod
    def store_transition(
        self,
        action: int,
        observation: np.ndarray,
        reward: float,
        new_observation: np.ndarray,
        done: bool,
        prev_action: Optional[int] = None,
    ) -> None:
        pass


class SimpleDQNAgent(DQNAgent):
    def __init__(
        self,
        observation_size: int,
        n_actions: int,
        config: TrainConfig,
        device,
        enable_dueling_dqn: bool,
    ):
        self._config = config
        self._replay_buffer = ReplayBuffer(
            buffer_size=config.max_mem_size,
            observation_size=config.observation_size,
        )
        self.epsilon = config.epsilon
        self._n_actions = n_actions
        if enable_dueling_dqn:
            self.policy_net = DuelingDQN(
                observation_size=observation_size,
                fc_dims=config.fc_dim,
                n_actions=self._n_actions,
            ).to(device)
            self.target_net = DuelingDQN(
                observation_size=observation_size,
                fc_dims=config.fc_dim,
                n_actions=self._n_actions,
            ).to(device)
        else:
            self.policy_net = DQN(
                observation_size=observation_size,
                fc_dims=config.fc_dim,
                n_actions=self._n_actions,
            ).to(device)
            self.target_net = DQN(
                observation_size=observation_size,
                fc_dims=config.fc_dim,
                n_actions=self._n_actions,
            ).to(device)
        self._actions_taken = []
        self._learn_step_counter = 0
        self._device = device

        self._optimizer = optim.Adam(self.policy_net.parameters(), lr=config.lr)
        self._loss = nn.MSELoss()
        self.policy_net.eval()
        self.target_net.eval()

    def get_epsilon(self) -> float:
        return self.epsilon

    def get_policy_net_state_dict(self) -> dict:
        return self.policy_net.state_dict()

    def episode_reset(self) -> None:
        self._actions_taken = []

    def episode_done(self) -> None:
        pass

    @torch.no_grad()
    def choose_action(
        self,
        observation: np.ndarray,
        enable_epsilon_greedy: bool,
        forbidden_actions: set[int],
        eval_mode: bool,
        hack: bool = False,
    ) -> tuple[int, float]:
        self.policy_net.eval()
        observation = observation.astype(np.float32)
        actions_q = self.policy_net(
            torch.tensor(observation, device=self._device)[None, ...]
        )
        assert len(self._config.actions) == actions_q.shape[-1]
        value = actions_q.max().item()
        actions_q = actions_q.squeeze()
        if eval_mode:
            if hack:
                while (
                    torch.argmax(actions_q).item() in self._actions_taken
                    and actions_q.max() > 0
                ):
                    actions_q[torch.argmax(actions_q).item()] = 0.0
            action = torch.argmax(actions_q).item()
            self._actions_taken.append(action)
            return action, value
        else:
            if forbidden_actions is None:
                forbidden_actions = set()
            if len(forbidden_actions) >= self._n_actions:
                print(
                    "Warning: all actions are forbidden, choosing a random action",
                    file=sys.stderr,
                )
                return (
                    np.random.choice(
                        list(set(range(self._n_actions)) - forbidden_actions)
                    ),
                    value,
                )
            if np.random.random() <= self.epsilon and enable_epsilon_greedy:
                action = np.random.choice(
                    list(set(range(self._n_actions)) - forbidden_actions)
                )
            else:
                allowed_actions = list(set(range(self._n_actions)) - forbidden_actions)
                action = allowed_actions[
                    torch.argmax(actions_q[allowed_actions]).item()
                ]

            self._actions_taken.append(action)
            return action, value

    def learn(self) -> Optional[float]:
        self.policy_net.train()
        self.target_net.eval()
        if (
            self._replay_buffer.get_ready_data_size()
            < self._config.learn_memory_threshold
        ):
            return
        self._optimizer.zero_grad()
        self._replace_target_network()
        dqn_batch = self._replay_buffer.get_batch(self._config.batch_size, self._device)
        q_current, q_target = self._get_q_current_and_target(dqn_batch)
        loss = self._loss(q_target, q_current).to(self._device)
        loss_val = loss.item()
        loss.backward()
        self._optimizer.step()
        self._learn_step_counter += 1
        if self.epsilon > self._config.epsilon_end:
            self.epsilon -= self._config.epsilon_dec
        else:
            self.epsilon = self._config.epsilon_end
        self.policy_net.eval()
        self.target_net.eval()
        return loss_val

    def store_transition(
        self,
        action: int,
        observation: np.ndarray,
        reward: float,
        new_observation: np.ndarray,
        done: bool,
        prev_action: Optional[int] = None,
    ) -> None:
        self._replay_buffer.store_transition(
            action, observation, reward, new_observation, done
        )

    def _replace_target_network(self) -> None:
        if self._config.enable_soft_update:
            target_net_state_dict = self.target_net.state_dict()
            policy_net_state_dict = self.policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = target_net_state_dict[
                    key
                ] * self._config.tau + policy_net_state_dict[key] * (
                    1 - self._config.tau
                )
            self.target_net.load_state_dict(target_net_state_dict)
        else:
            if self._learn_step_counter % self._config.replace_period == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    def _get_q_current_and_target(
        self, dqn_batch: DQNTrainBatch
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_index = np.arange(dqn_batch.batch_size, dtype=np.int32)
        q_values_current = self.policy_net.forward(dqn_batch.state_batch)
        q_current = q_values_current[batch_index, dqn_batch.action_batch]

        with torch.no_grad():
            q_next = self.target_net.forward(dqn_batch.new_state_batch).max(dim=1)[0]
            q_next[dqn_batch.terminal_batch] = 0.0
            q_target = dqn_batch.reward_batch + self._config.gamma * q_next
        assert (
            q_target.shape == q_current.shape
            and len(q_target.shape) == 1
            and q_target.shape[0] == self._config.batch_size
        )
        return q_current, q_target


class DoubleDQNAgent(SimpleDQNAgent):
    def _get_q_current_and_target(
        self, dqn_batch: DQNTrainBatch
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_index = np.arange(dqn_batch.batch_size, dtype=np.int32)
        q_values_current = self.policy_net.forward(dqn_batch.state_batch)
        q_current = q_values_current[batch_index, dqn_batch.action_batch]

        with torch.no_grad():
            q_new_state = self.policy_net.forward(dqn_batch.new_state_batch)
            q_current_new_state_argmax = torch.argmax(q_new_state, dim=-1)
            q_next = self.target_net.forward(dqn_batch.new_state_batch)[
                batch_index, q_current_new_state_argmax
            ]
        q_next[dqn_batch.terminal_batch] = 0.0
        q_target = dqn_batch.reward_batch + self._config.gamma * q_next
        assert (
            q_target.shape == q_current.shape
            and len(q_target.shape) == 1
            and q_target.shape[0] == self._config.batch_size
        )
        return q_current, q_target


class _TwinDQNSubAgent:
    def __init__(
        self,
        observation_size: int,
        n_actions: int,
        config: TrainConfig,
        device,
        enable_dueling_dqn: bool,
    ):
        self._config = config
        self._replay_buffer = ReplayBuffer(
            buffer_size=config.max_mem_size,
            observation_size=config.observation_size,
        )
        self.epsilon = config.epsilon
        self._n_actions = n_actions
        if enable_dueling_dqn:
            self.policy_net = DuelingDQN(
                observation_size=observation_size,
                fc_dims=config.fc_dim,
                n_actions=self._n_actions,
            ).to(device)
        else:
            self.policy_net = DQN(
                observation_size=observation_size,
                fc_dims=config.fc_dim,
                n_actions=self._n_actions,
            ).to(device)
        self._actions_taken = []
        self._learn_step_counter = 0
        self._device = device
        self._optimizer = optim.Adam(self.policy_net.parameters(), lr=config.lr)
        self._loss = nn.MSELoss()
        self.policy_net.eval()

    def episode_reset(self) -> None:
        self._actions_taken = []

    def episode_done(self) -> None:
        pass

    @torch.no_grad()
    def choose_action(
        self,
        observation: np.ndarray,
        enable_epsilon_greedy: bool,
        forbidden_actions: set[int],
        eval_mode: bool,
        hack: bool = False,
    ) -> tuple[int, float]:
        observation = observation.astype(np.float32)
        actions_q = self.policy_net(
            torch.tensor(observation, device=self._device)[None, ...]
        )
        value = actions_q.max().item()
        actions_q = actions_q.squeeze()
        assert len(self._config.actions) == actions_q.shape[-1]
        if eval_mode:
            while (
                torch.argmax(actions_q).item() in self._actions_taken
                and actions_q.max() > 0
            ):
                actions_q[torch.argmax(actions_q).item()] = 0.0
            action = torch.argmax(actions_q).item()
            self._actions_taken.append(action)
            return action, value
        else:
            if forbidden_actions is None:
                forbidden_actions = set()
            if len(forbidden_actions) >= self._n_actions:
                print(
                    "Warning: all actions are forbidden, choosing a random action",
                    file=sys.stderr,
                )
                return (
                    np.random.choice(
                        list(set(range(self._n_actions)) - forbidden_actions)
                    ),
                    value,
                )
            if np.random.random() <= self.epsilon and enable_epsilon_greedy:
                action = np.random.choice(
                    list(set(range(self._n_actions)) - forbidden_actions)
                )
            else:
                allowed_actions = list(set(range(self._n_actions)) - forbidden_actions)
                action = allowed_actions[
                    torch.argmax(actions_q[allowed_actions]).item()
                ]

            self._actions_taken.append(action)
            return action, value

    def store_transition(
        self,
        action: int,
        observation: np.ndarray,
        reward: float,
        new_observation: np.ndarray,
        done: bool,
        prev_action: Optional[int] = None,
    ) -> None:
        self._replay_buffer.store_transition(
            action, observation, reward, new_observation, done
        )

    def is_ready_for_train(self) -> bool:
        return (
            self._replay_buffer.get_ready_data_size()
            >= self._config.learn_memory_threshold
        )

    def learn(self, twin_net) -> Optional[float]:
        self.policy_net.train()
        twin_net.eval()
        if (
            self._replay_buffer.get_ready_data_size()
            < self._config.learn_memory_threshold
        ):
            return
        self._optimizer.zero_grad()
        dqn_batch = self._replay_buffer.get_batch(self._config.batch_size, self._device)
        q_current, q_target = self._get_q_current_and_target(dqn_batch, twin_net)
        loss = self._loss(q_target, q_current).to(self._device)
        loss_val = loss.item()
        loss.backward()
        self._optimizer.step()
        self._learn_step_counter += 1
        if self.epsilon > self._config.epsilon_end:
            self.epsilon -= self._config.epsilon_dec
        else:
            self.epsilon = self._config.epsilon_end
        self.policy_net.eval()
        twin_net.eval()
        return loss_val

    def _get_q_current_and_target(
        self, dqn_batch: DQNTrainBatch, twin_net: nn.Module
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_index = np.arange(dqn_batch.batch_size, dtype=np.int32)
        q_values_current = self.policy_net.forward(dqn_batch.state_batch)
        q_current = q_values_current[batch_index, dqn_batch.action_batch]

        q_values_twin = twin_net.forward(dqn_batch.new_state_batch)
        q_twin_argmax = torch.argmax(q_values_twin, dim=1)
        with torch.no_grad():
            q_next = torch.min(
                torch.stack(
                    [
                        twin_net.forward(dqn_batch.new_state_batch)[
                            batch_index, q_twin_argmax
                        ],
                        self.policy_net.forward(dqn_batch.new_state_batch)[
                            batch_index, q_twin_argmax
                        ],
                    ],
                    dim=-1,
                ),
                dim=-1,
            )[0]
        q_next[dqn_batch.terminal_batch] = 0.0
        q_target = dqn_batch.reward_batch + self._config.gamma * q_next
        assert (
            q_target.shape == q_current.shape
            and len(q_target.shape) == 1
            and q_target.shape[0] == self._config.batch_size
        )
        return q_current, q_target


class TwinDQNAgent(DQNAgent):
    def __init__(
        self,
        observation_size: int,
        n_actions: int,
        config: TrainConfig,
        device,
        enable_dueling_dqn: bool,
    ):
        self._config = config
        self.tmp_net = DQN(
            observation_size=observation_size,
            fc_dims=config.fc_dim,
            n_actions=n_actions,
        ).to(device)
        self._agent_1 = _TwinDQNSubAgent(
            observation_size, n_actions, config, device, enable_dueling_dqn
        )
        self._agent_2 = _TwinDQNSubAgent(
            observation_size, n_actions, config, device, enable_dueling_dqn
        )
        self._cur_agent = self._agent_1

    def get_epsilon(self) -> float:
        return self._agent_1.epsilon

    def get_policy_net_state_dict(self) -> dict:
        return self._agent_1.policy_net.state_dict()

    def episode_reset(self):
        if self._cur_agent is self._agent_1:
            self._cur_agent = self._agent_2
        elif self._cur_agent is self._agent_2:
            self._cur_agent = self._agent_1
        self._cur_agent.episode_reset()

    def episode_done(self) -> None:
        self._cur_agent.episode_done()

    @torch.no_grad()
    def choose_action(
        self,
        observation: np.ndarray,
        enable_epsilon_greedy: bool,
        forbidden_actions: set[int],
        eval_mode: bool,
        hack: bool = False,
    ) -> tuple[int, float]:
        return self._cur_agent.choose_action(
            observation, enable_epsilon_greedy, forbidden_actions, eval_mode
        )

    def learn(self) -> Optional[float]:
        if (
            not self._agent_1.is_ready_for_train()
            or not self._agent_2.is_ready_for_train()
        ):
            return
        self.tmp_net.load_state_dict(self._agent_1.policy_net.state_dict())
        self.tmp_net.eval()
        loss_val = self._agent_1.learn(self._agent_2.policy_net)
        self._agent_2.learn(self.tmp_net)
        return loss_val

    def store_transition(
        self,
        action: int,
        observation: np.ndarray,
        reward: float,
        new_observation: np.ndarray,
        done: bool,
        prev_action: Optional[int] = None,
    ) -> None:
        self._cur_agent.store_transition(
            action, observation, reward, new_observation, done
        )


class LstmDQNAgent(DQNAgent):
    def __init__(
        self, observation_size: int, n_actions: int, config: TrainConfig, device
    ):
        self._config = config
        self._replay_buffer = ReplayBufferForLSTM(
            buffer_size=config.max_mem_size,
            observation_size=config.observation_size,
            episode_length=config.episode_length,
        )
        self.epsilon = config.epsilon
        self._n_actions = n_actions
        self.policy_net = DQNLSTM(
            observation_size=observation_size,
            fc_dims=config.fc_dim,
            hidden_size=config.lstm_hidden_size,
            n_actions=self._n_actions,
        ).to(device)
        self.target_net = DQNLSTM(
            observation_size=observation_size,
            fc_dims=config.fc_dim,
            hidden_size=config.lstm_hidden_size,
            n_actions=self._n_actions,
        ).to(device)
        self._actions_taken = []
        self._learn_step_counter = 0
        self._device = device
        self.h_prev = None
        self.c_prev = None
        self.prev_action = 0

        self._optimizer = optim.Adam(self.policy_net.parameters(), lr=config.lr)
        self._loss = nn.HuberLoss()

    def get_epsilon(self) -> float:
        return self.epsilon

    def get_policy_net_state_dict(self) -> dict:
        return self.policy_net.state_dict()

    def episode_reset(self):
        self._actions_taken = []
        self.h_prev = None
        self.c_prev = None
        self.prev_action = 0
        self._replay_buffer.episode_reset()

    def episode_done(self) -> None:
        self._replay_buffer.episode_done()

    @torch.no_grad()
    def choose_action(
        self,
        observation: np.ndarray,
        enable_epsilon_greedy: bool,
        forbidden_actions: set[int],
        eval_mode: bool,
        hack: bool = False,
    ) -> tuple[int, float]:
        observation = observation.astype(np.float32)
        actions_q, self.h_prev, self.c_prev = self.policy_net.forward_step(
            torch.tensor(observation, device=self._device),
            self.prev_action,
            h_prev=self.h_prev,
            c_prev=self.c_prev,
        )
        actions_q = actions_q.squeeze()
        assert len(self._config.actions) == actions_q.shape[-1]
        value = actions_q.max().item()
        if eval_mode:
            if hack:
                while (
                    torch.argmax(actions_q).item() in self._actions_taken
                    and actions_q.max() > 0
                ):
                    actions_q[torch.argmax(actions_q).item()] = 0.0
            action = torch.argmax(actions_q).item()
            self._actions_taken.append(action)
            return action, value
        else:
            if forbidden_actions is None:
                forbidden_actions = set()
            if len(forbidden_actions) >= self._n_actions:
                print(
                    "Warning: all actions are forbidden, choosing a random action",
                    file=sys.stderr,
                )
                return (
                    np.random.choice(
                        list(set(range(self._n_actions)) - forbidden_actions)
                    ),
                    value,
                )
            if np.random.random() <= self.epsilon and enable_epsilon_greedy:
                action = np.random.choice(
                    list(set(range(self._n_actions)) - forbidden_actions)
                )
            else:
                allowed_actions = list(set(range(self._n_actions)) - forbidden_actions)
                action = allowed_actions[
                    torch.argmax(actions_q[allowed_actions]).item()
                ]
            self._actions_taken.append(action)
            self.prev_action = action
            return action, value

    def learn(self) -> Optional[float]:
        self.policy_net.train()
        self.target_net.eval()
        if (
            self._replay_buffer.get_ready_data_size()
            < self._config.learn_memory_threshold
        ):
            return
        self._optimizer.zero_grad()
        self._replace_target_network()
        dqn_batch = self._replay_buffer.get_batch(self._config.batch_size, self._device)
        q_current, q_target = self._get_q_current_and_target(dqn_batch)
        loss = self._loss(q_target, q_current).to(self._device)
        loss_val = loss.item()
        loss.backward()
        self._optimizer.step()
        self._learn_step_counter += 1
        if self.epsilon > self._config.epsilon_end:
            self.epsilon -= self._config.epsilon_dec
        else:
            self.epsilon = self._config.epsilon_end
        self.policy_net.eval()
        self.target_net.eval()
        return loss_val

    def store_transition(
        self,
        action: int,
        observation: np.ndarray,
        reward: float,
        new_observation: np.ndarray,
        done: bool,
        prev_action: Optional[int] = None,
    ) -> None:
        if prev_action is None:
            raise ValueError("LstmDQNAgent: prev_action not provided")
        self._replay_buffer.store_transition(
            prev_action, action, observation, reward, new_observation, done
        )

    def _replace_target_network(self) -> None:
        if self._config.enable_soft_update:
            target_net_state_dict = self.target_net.state_dict()
            policy_net_state_dict = self.policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = target_net_state_dict[
                    key
                ] * self._config.tau + policy_net_state_dict[key] * (
                    1 - self._config.tau
                )
            self.target_net.load_state_dict(target_net_state_dict)
        else:
            if self._learn_step_counter % self._config.replace_period == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    def _get_q_current_and_target(
        self, dqn_batch: DQNTrainBatch
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_index = np.arange(dqn_batch.batch_size, dtype=np.int32)
        q_current = self.policy_net.forward(
            dqn_batch.state_batch,
            dqn_batch.prev_action_batch,
            dqn_batch.lengths,
        )[batch_index, dqn_batch.final_action_batch]
        with torch.no_grad():
            q_next = self.target_net.forward(
                torch.cat(
                    [dqn_batch.state_batch[..., [0], :], dqn_batch.new_state_batch],
                    dim=-2,
                ),
                torch.cat(
                    [dqn_batch.prev_action_batch[..., [0]], dqn_batch.action_batch],
                    dim=-1,
                ),
                dqn_batch.lengths + 1,
            ).max(dim=1)[0]
        q_next[dqn_batch.terminal_batch] = 0.0
        q_target = dqn_batch.reward_batch + self._config.gamma * q_next
        q_current = q_current.view(-1)
        q_target = q_target.view(-1)
        assert (
            q_target.shape == q_current.shape
            and len(q_target.shape) == 1
            and q_target.shape[0] == self._config.batch_size
        )
        return q_current, q_target
