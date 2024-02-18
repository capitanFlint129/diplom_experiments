import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class DQN(nn.Module):
    def __init__(
        self,
        observation_size: int,
        fc_dims: int,
        n_actions: int,
    ):
        super(DQN, self).__init__()
        self.observation_size = observation_size

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
            fc_dims=config["fc_dim"],
            n_actions=self.n_actions,
        )
        self.Q_next = DQN(
            observation_size=observation_size,
            fc_dims=config["fc_dim"],
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

        self.optimizer = optim.Adam(self.Q_eval.parameters(), lr=config["alpha"])
        # todo : try huber loss
        self.loss = nn.SmoothL1Loss()

    def episode_reset(self):
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
    def choose_action(self, observation, enable_epsilon_greedy: bool = True):
        observation = observation.astype(np.float32)
        actions_q = self.Q_eval(
            torch.tensor(observation, device=self.device)[None, ...]
        )
        action = torch.argmax(actions_q).item()
        if np.random.random() <= self.epsilon and enable_epsilon_greedy:
            action = np.random.choice(self.action_space)
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
        with torch.no_grad():
            q_next = self.Q_next.forward(new_state_batch).max(dim=1)[0]
        # if and index of the batch is done (True), then set next reward to 0
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
