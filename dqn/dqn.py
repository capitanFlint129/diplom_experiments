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
            nn.Linear(fc_dims, fc_dims),
            nn.Linear(fc_dims, fc_dims),
        )
        self.rnn_encoder = nn.LSTM(fc_dims, hidden_size, batch_first=True)
        self.output_net = nn.Sequential(
            nn.Linear(hidden_size, fc_dims),
            nn.Linear(fc_dims, fc_dims),
            nn.Linear(fc_dims, fc_dims),
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
            and output.shape[0] == observation.shape[2] == self.hidden_size
        ), f"{output.shape}"
        final_outputs = torch.index_select(output, 1, sequence_lengths - 1)
        actions_q = self.output_net(final_outputs)
        return actions_q

    def forward_step(
        self,
        observation: torch.Tensor,
        prev_action: torch.LongTensor,
        h_prev: Optional[torch.Tensor] = None,
        c_prev: Optional[torch.Tensor] = None,
    ) -> tuple[Any, Any, Any]:
        assert observation.shape == [1, self.observation_size], f"{observation.shape}"
        assert prev_action.shape == [1]
        prev_action_emb = self.action_embedding(prev_action[None, ...])
        input = torch.cat((observation, prev_action_emb), dim=-1)
        rnn_input = self.input_net(input)
        if h_prev is None or c_prev is None:
            output, (hn, cn) = self.rnn_encoder(rnn_input)
        else:
            output, (hn, cn) = self.rnn_encoder(rnn_input, (h_prev, c_prev))
        actions_probabilities = self.output_net(output)
        return actions_probabilities, hn, cn


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
        self.reward_mem = np.zeros(self.max_mem_size, dtype=np.float32)
        self.terminal_mem = np.zeros(self.max_mem_size, dtype=bool)
        self.learn_step_counter = 0
        self.device = device
        self.h_prev = None
        self.c_prev = None
        self.prev_action = None
        self.to(self.device)

        self.optimizer = optim.Adam(self.Q_eval.parameters(), lr=config["alpha"])
        # todo : try huber loss
        self.loss = nn.SmoothL1Loss()

    def episode_reset(self):
        self.h_prev = None
        self.c_prev = None
        self.prev_action = torch.zeros(
            self.Q_eval.action_embedding_size,
            device=self.device,
        )
        self.actions_taken = []

    def store_transition(self, action, state, reward, new_state, done):
        index = self.mem_cntr % self.max_mem_size
        self.state_mem[index] = state
        self.new_state_mem[index] = new_state
        self.action_mem[index] = action
        self.reward_mem[index] = reward
        self.terminal_mem[index] = done
        self.mem_cntr += 1

    @torch.no_grad()
    def choose_action(self, observation, disable_epsilon_greedy=False):
        observation = observation.astype(np.float32)
        state = torch.tensor(observation, device=self.device)[None, ...]
        actions_q, self.h_prev, self.c_prev = self.Q_eval.forward_step(
            state, self.prev_action
        )
        # network seems to choose same action over and over, even with zero reward,
        # trying giving negative reward for choosing same action multiple times
        # while torch.argmax(actions).item() in self.actions_taken:
        #     actions[0][torch.argmax(actions).item()] = 0.0
        action = torch.argmax(actions_q).item()
        if not (np.random.random() > self.epsilon or disable_epsilon_greedy):
            action = np.random.choice(self.action_space)
        self.actions_taken.append(action)
        self.prev_action = action
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
        # have to calculate scalar of importance so that we don't update network in a biased way
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        # sending a batch of states to device
        state_batch = torch.tensor(self.state_mem[batch], device=self.device)
        new_state_batch = torch.tensor(self.new_state_mem[batch], device=self.device)
        reward_batch = torch.tensor(self.reward_mem[batch], device=self.device)
        terminal_batch = torch.tensor(self.terminal_mem[batch], device=self.device)
        action_batch = self.action_mem[batch]
        """
		calling forward with a batch of states gives us a batch of Q-values.
		The batch_index just selects each group of Q-values and action_batch
		selects the action we took in each group of Q-values.
		We use batch_index here instead of batch because order no longer
		matters after passing through the network. Ex.) a batch of [22,74,3,43]
		would select those respective states from the state memory, and pass them through
		the network, but after they are passed though we are indexing based on the size of
		the batch, not the replay buffer.
		"""
        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_next.forward(new_state_batch).max(dim=1)[0]
        # if and index of the batch is done (True), then set next reward to 0
        q_next[terminal_batch] = 0.0
        q_target = reward_batch + self.gamma * q_next
        loss = self.Q_eval.loss(q_target, q_eval).to(self.device)
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
    if end_index < begin_index:
        end_index += array_size
    indexes = np.arange(begin_index, end_index) % array_size
    return indexes


def _pad_seq_to_len(seq: np.ndarray, seq_len: int) -> np.ndarray:
    pad_width = tuple([(0, seq_len - seq.shape[0])] + [(0, 0)] * (len(seq.shape) - 1))
    return np.pad(seq, pad_width)
