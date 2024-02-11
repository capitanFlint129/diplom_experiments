import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        alpha: float,
        observation_size: int,
        action_embedding_size: int,
        hidden_size: int,
        fc_dims: int,
        n_actions: int,
    ):
        super(DQN, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.fc3_dims)
        self.fc4 = nn.Linear(self.fc3_dims, self.n_actions)
        self.hidden_size = hidden_size
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
        self.softmax = nn.Softmax(dim=-1)
        self.optimizer = optim.Adam(self.parameters(), lr=ALPHA)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.loss = nn.SmoothL1Loss()  # try huber loss

    def forward(
        self,
        observation: torch.Tensor,
        action: torch.LongTensor,
        sequence_lengths: torch.LongTensor,
    ) -> torch.LongTensor:
        assert (
            observation.shape == action.shape and len(action.shape) == 3
        ), f"{observation.shape} - {action.shape}"
        action_embs = self.action_embedding(action)
        input = torch.cat((observation, action_embs), dim=-1)
        rnn_input = self.input_net(input)
        output, (hn, cn) = self.rnn_encoder(rnn_input)
        assert (
            len(output.shape) == 3
            and output.shape[0] == observation.shape[0]
            and output.shape[1] == observation.shape[1]
            and output.shape[0] == observation.shape[2] == self.hidden_size
        ), f"{output.shape}"
        final_outputs = torch.index_select(output, 1, sequence_lengths - 1)
        actions_probabilities = self.output_net(final_outputs)
        return actions_probabilities


class Agent(nn.Module):
    def __init__(self, input_dims, n_actions, config, device):
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
            config["alpha"],
            input_dims,
            fc1_dims=config["fc_dim"],
            fc2_dims=config["fc_dim"],
            fc3_dims=config["fc_dim"],
            n_actions=self.n_actions,
        )
        self.Q_next = DQN(
            config["alpha"],
            input_dims,
            fc1_dims=config["fc_dim"],
            fc2_dims=config["fc_dim"],
            fc3_dims=config["fc_dim"],
            n_actions=self.n_actions,
        )
        self.actions_taken = []
        # star unpacks list into positional arguments
        self.state_mem = np.zeros((self.max_mem_size, *input_dims), dtype=np.float32)
        self.new_state_mem = np.zeros(
            (self.max_mem_size, *input_dims), dtype=np.float32
        )
        self.action_mem = np.zeros(self.max_mem_size, dtype=np.int32)
        self.reward_mem = np.zeros(self.max_mem_size, dtype=np.float32)
        self.terminal_mem = np.zeros(self.max_mem_size, dtype=bool)
        self.learn_step_counter = 0
        self.device = device
        self.to(self.device)

    def store_transition(self, action, state, reward, new_state, done):
        # what is the position of the first unoccupied memory
        index = self.mem_cntr % self.max_mem_size
        self.state_mem[index] = state
        self.new_state_mem[index] = new_state
        self.action_mem[index] = action
        self.reward_mem[index] = reward
        self.terminal_mem[index] = done
        self.mem_cntr += 1

    @torch.no_grad()
    def choose_action(self, observation, disable_epsilon_greedy=False):
        if np.random.random() > self.epsilon or disable_epsilon_greedy:
            # sends observation as tensor to device
            # convert to float - > compiler gyms autophase vector is a long
            observation = observation.astype(np.float32)
            state = torch.tensor(observation, device=self.device)[None, ...]
            actions_probabilities = self.Q_eval.forward(state).squeeze()
            # network seems to choose same action over and over, even with zero reward,
            # trying giving negative reward for choosing same action multiple times
            # while torch.argmax(actions).item() in self.actions_taken:
            #     # TODO здесь опять не работает
            #     actions[0][torch.argmax(actions).item()] = 0.0
            action = torch.argmax(actions_probabilities).item()
            # action = np.random.choice(
            #     actions_probabilities.shape[0],
            #     p=actions_probabilities.detach().cpu().numpy(),
            # )
            self.actions_taken.append(action)
        else:
            action = np.random.choice(self.action_space)
        return action

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.Q_next.load_state_dict(self.Q_eval.state_dict())

    def learn(self):
        # start learning as soon as batch size of memory is filled
        if self.mem_cntr < self.learn_memory_threshold:
            return
        # set gradients to zero
        self.Q_eval.optimizer.zero_grad()
        self.replace_target_network()
        # select subset of memorys
        max_mem = min(self.mem_cntr, self.max_mem_size)
        # take a selection of the size of the batch size from the current pool of memory's
        # pool of memory's will be full by the time we get here
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
        self.Q_eval.optimizer.step()
        self.learn_step_counter += 1

        if self.epsilon > self.eps_end:
            self.epsilon -= self.eps_dec
        else:
            self.epsilon = self.eps_end
        return loss_val
