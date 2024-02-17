from typing import Optional, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Start implementing ideas from Deep RL Bootcamp series on youtube

"""
- concatentate observation with previous observations (they used 4)
- use huber loss (same from [-1,1], but penalizes less for larger errors)
- use RMSProp instead of grad descent
- add more exploration at the beginning
- could try prioritized experience replay again...
- could also try dueling _dqn to see the effect of the advantage property

"""


class DQN(nn.Module):
    def __init__(
        self,
        observation_size: int,
        action_embedding_size: int,
        hidden_size: int,
        fc_dims: int,
        n_actions: int,
    ):
        super(DQN, self).__init__()
        self.hidden_size = hidden_size
        self.observation_size = observation_size
        self.action_embedding_size = action_embedding_size

        self.action_embedding = nn.Embedding(n_actions, action_embedding_size)
        self.input_net = nn.Sequential(
            nn.Linear(action_embedding_size + observation_size, fc_dims),
            nn.ReLU(),
            nn.Linear(fc_dims, fc_dims),
            nn.ReLU(),
            nn.Linear(fc_dims, fc_dims),
        )
        self.rnn_encoder = nn.LSTM(fc_dims, hidden_size, batch_first=True)
        self.output_net = nn.Sequential(
            nn.Linear(hidden_size, fc_dims),
            nn.ReLU(),
            nn.Linear(fc_dims, fc_dims),
            nn.ReLU(),
            nn.Linear(fc_dims, n_actions),
        )

    def forward(
        self,
        observation: torch.Tensor,
        prev_action: torch.Tensor,
        sequence_lengths: torch.Tensor,
    ) -> torch.LongTensor:
        assert (
            observation.shape == prev_action.shape and len(prev_action.shape) == 3
        ), f"{observation.shape} - {prev_action.shape}"
        action_embs = self.action_embedding(prev_action)
        input = torch.cat((observation, action_embs), dim=-1)
        rnn_input = self.input_net(input)
        output, (_, _) = self.rnn_encoder(rnn_input)
        assert (
            len(output.shape) == 3
            and output.shape[0] == observation.shape[0]
            and output.shape[1] == observation.shape[1]
            and output.shape[2] == self.hidden_size
        ), f"{output.shape}"
        output_t = output.transpose(0, 1)
        masks = sequence_lengths.view(1, -1, 1).expand(
            sequence_lengths.max().item() + 1, output_t.size(1), output_t.size(2)
        )
        final_outputs = output_t.gather(0, masks)[0]
        actions_q = self.output_net(final_outputs)
        return actions_q

    def forward_step(
        self,
        observation: torch.Tensor,
        prev_action: Optional[int],
        h_prev: Optional[torch.Tensor] = None,
        c_prev: Optional[torch.Tensor] = None,
    ) -> tuple[Any, Any, Any]:
        assert (
            len(observation.shape) == 2
            and observation.shape[0] == 1
            and observation.shape[1] == self.observation_size
        ), observation.shape
        if prev_action is None:
            prev_action_emb = torch.zeros(
                1, self.action_embedding_size, device=observation.device
            )
        else:
            prev_action_emb = self.action_embedding(
                torch.tensor([prev_action], device=observation.device)
            )
        input = torch.cat((observation, prev_action_emb), dim=-1)
        rnn_input = self.input_net(input)
        if h_prev is None or c_prev is None:
            output, (hn, cn) = self.rnn_encoder(rnn_input)
        else:
            output, (hn, cn) = self.rnn_encoder(rnn_input, (h_prev, c_prev))
        actions_q = self.output_net(output)
        return actions_q, hn, cn


class Agent(nn.Module):
    def __init__(self, observation_size, n_actions, config, device):
        super(Agent, self).__init__()
        self.eps_dec = config["epsilon_dec"]
        self.eps_end = config["epsilon_end"]
        self.max_mem_size = config["max_mem_size"]
        self.replace_target_cnt = config["replace"]
        self.gamma = config["gamma"]
        self.epsilon = config["epsilon"]
        self.eps_end = config["epsilon_end"]
        self.eps_dec = config["epsilon_dec"]
        self.n_actions = n_actions
        self.action_space = [i for i in range(n_actions)]
        self.max_mem_size = config["max_mem_size"]
        self.batch_size = config["batch_size"]
        self.learn_memory_threshold = config["learn_memory_threshold"]
        # keep track of position of first available memory
        self.mem_cntr = 0
        self.Q_eval = DQN(
            observation_size=observation_size,
            action_embedding_size=config["action_embedding_size"],
            hidden_size=config["lstm_hidden_size"],
            fc_dims=config["fc_dim"],
            n_actions=self.n_actions,
        )
        self.Q_next = DQN(
            observation_size=observation_size,
            action_embedding_size=config["action_embedding_size"],
            hidden_size=config["lstm_hidden_size"],
            fc_dims=config["fc_dim"],
            n_actions=self.n_actions,
        )
        self.actions_taken = []
        # star unpacks list into positional arguments
        self.state_mem = np.zeros(
            (self.max_mem_size, observation_size), dtype=np.float32
        )
        self.new_state_mem = np.zeros(
            (self.max_mem_size, observation_size), dtype=np.float32
        )
        self.action_mem = np.zeros(self.max_mem_size, dtype=np.int32)
        self.prev_action_mem = np.zeros(self.max_mem_size, dtype=np.int32)
        self.reward_mem = np.zeros(self.max_mem_size, dtype=np.float32)
        self.terminal_mem = np.zeros(self.max_mem_size, dtype=bool)
        self.episode_start_mem = np.zeros(self.max_mem_size, dtype=np.int32)
        self.learn_step_counter = 0
        self.device = device
        self.h_prev = None
        self.c_prev = None
        self.prev_action = None
        self.end_index = -1
        self.to(self.device)

        self.optimizer = optim.Adam(self.Q_eval.parameters(), lr=config["alpha"])
        # todo : try huber loss
        self.loss = nn.SmoothL1Loss()

    def episode_reset(self):
        self.h_prev = None
        self.c_prev = None
        self.prev_action = None
        self.actions_taken = []
        self.end_index = self.mem_cntr % self.max_mem_size

    def episode_done(self):
        begin_index = (self.mem_cntr - 1) % self.max_mem_size
        indexes = _get_range_for_cyclic(begin_index, self.end_index, self.max_mem_size)
        rev_indexes = indexes[::-1]
        self.state_mem[indexes] = self.state_mem[rev_indexes]
        self.new_state_mem[indexes] = self.new_state_mem[rev_indexes]
        self.action_mem[indexes] = self.action_mem[rev_indexes]
        self.prev_action_mem[indexes] = self.prev_action_mem[rev_indexes]
        self.reward_mem[indexes] = self.reward_mem[rev_indexes]
        self.terminal_mem[indexes] = self.terminal_mem[rev_indexes]
        self.episode_start_mem[indexes] = begin_index

    def store_transition(self, prev_action, action, state, reward, new_state, done):
        index = self.mem_cntr % self.max_mem_size
        self.state_mem[index] = state
        self.new_state_mem[index] = new_state
        self.action_mem[index] = action
        self.prev_action_mem[index] = prev_action
        self.reward_mem[index] = reward
        self.terminal_mem[index] = done
        self.mem_cntr += 1

    @torch.no_grad()
    def choose_action(self, observation, enable_epsilon_greedy=True):
        observation = observation.astype(np.float32)
        actions_q, self.h_prev, self.c_prev = self.Q_eval.forward_step(
            torch.tensor(observation, device=self.device)[None, ...],
            self.prev_action,
        )
        # network seems to choose same action over and over, even with zero reward,
        # trying giving negative reward for choosing same action multiple times
        # while torch.argmax(actions).item() in self.actions_taken:
        #     actions[0][torch.argmax(actions).item()] = 0.0
        action = torch.argmax(actions_q).item()
        if np.random.random() <= self.epsilon and enable_epsilon_greedy:
            action = np.random.choice(self.action_space)
        self.actions_taken.append(action)
        self.prev_action = action
        return action

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.Q_next.load_state_dict(self.Q_eval.state_dict())
            self.Q_next.eval()

    def learn(self):
        mem_size = self.end_index
        if mem_size < self.learn_memory_threshold:
            return
        self.optimizer.zero_grad()
        self.replace_target_network()
        max_mem = min(mem_size, self.max_mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        state_batch = []
        new_state_batch = []
        reward_batch = []
        terminal_batch = []
        action_batch = []
        prev_action_batch = []
        final_action_batch = []
        max_len = 0
        lengths = []
        for end_index in batch:
            lengths.append(abs(self.episode_start_mem[end_index] - end_index + 1))
            max_len = max(lengths[-1], max_len)
        for end_index in batch:
            indexes = _get_range_for_cyclic(
                self.episode_start_mem[end_index], end_index, self.max_mem_size
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
        state_batch = torch.tensor(np.stack(state_batch), device=self.device)
        new_state_batch = torch.tensor(np.stack(new_state_batch), device=self.device)
        reward_batch = torch.tensor(reward_batch, device=self.device)
        terminal_batch = torch.tensor(terminal_batch, device=self.device)
        action_batch = torch.tensor(np.stack(action_batch), device=self.device)
        prev_action_batch = torch.tensor(
            np.stack(prev_action_batch), device=self.device
        )
        lengths = torch.tensor(lengths, device=self.device)
        final_action_batch = torch.tensor(final_action_batch, device=self.device)

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        q_eval = self.Q_eval.forward(state_batch, prev_action_batch, lengths - 1)[
            batch_index, final_action_batch
        ]
        with torch.no_grad():
            q_next = self.Q_next.forward(
                torch.cat([state_batch[..., [0], :], new_state_batch], dim=-2),
                torch.cat([prev_action_batch[..., [0]], action_batch], dim=-1),
                lengths,
            ).max(dim=1)[0]
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
