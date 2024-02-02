import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from absl import flags


class DQN(nn.Module):
    def __init__(self, ALPHA, input_dims, fc1_dims, fc2_dims, fc3_dims, n_actions):
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
        self.optimizer = optim.Adam(self.parameters(), lr=ALPHA)
        self.loss = nn.SmoothL1Loss()  # try huber loss
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        # first fully connected layer takes the state in as input, pass that output as input to activation function
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = self.fc4(x)

        return actions


class Agent:
    def __init__(self, input_dims, n_actions, config, device):
        self.learn_param = config["learn"]
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
        # keep track of position of first available memory
        self.mem_cntr = 0
        self.policy_net = DQN(
            config["alpha"],
            input_dims,
            fc1_dims=config["fc_dim"],
            fc2_dims=config["fc_dim"],
            fc3_dims=config["fc_dim"],
            n_actions=self.n_actions,
        ).to(device)
        self.target_net = DQN(
            config["alpha"],
            input_dims,
            fc1_dims=config["fc_dim"],
            fc2_dims=config["fc_dim"],
            fc3_dims=config["fc_dim"],
            n_actions=self.n_actions,
        ).to(device)
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

    def store_transition(self, action, state, reward, new_state, done):
        # what is the position of the first unoccupied memory
        index = self.mem_cntr % self.max_mem_size
        self.state_mem[index] = state
        self.new_state_mem[index] = new_state
        self.action_mem[index] = action
        self.reward_mem[index] = reward
        self.terminal_mem[index] = done

        self.mem_cntr += 1

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            # sends observation as tensor to device
            # convert to float - > compiler gyms autophase vector is a long
            observation = observation.astype(np.float32)
            state = torch.tensor([observation]).to(self.policy_net.device)
            actions = self.policy_net.forward(state)
            # network seems to choose same action over and over, even with zero reward,
            # trying giving negative reward for choosing same action multiple times
            while torch.argmax(actions).item() in self.actions_taken:
                actions[0][torch.argmax(actions).item()] = 0.0

            action = torch.argmax(actions).item()

            self.actions_taken.append(action)

        else:
            # take random action
            action = np.random.choice(self.action_space)

        return action

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def learn(self):
        # start learning as soon as batch size of memory is filled
        if self.mem_cntr < self.learn_param:
            return
        # set gradients to zero
        self.policy_net.optimizer.zero_grad()
        self.replace_target_network()
        # select subset of memorys
        max_mem = min(self.mem_cntr, self.max_mem_size)
        # take a selection of the size of the batch size from the current pool of memory's
        # pool of memory's will be full by the time we get here
        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        # have to calculate scalar of importance so that we don't update network in a biased way
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        # sending a batch of states to device
        state_batch = torch.tensor(self.state_mem[batch]).to(self.policy_net.device)
        new_state_batch = torch.tensor(self.new_state_mem[batch]).to(
            self.policy_net.device
        )
        reward_batch = torch.tensor(self.reward_mem[batch]).to(self.policy_net.device)
        terminal_batch = torch.tensor(self.terminal_mem[batch]).to(
            self.policy_net.device
        )
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
        policy_net = self.policy_net.forward(state_batch)[batch_index, action_batch]
        target_net = self.target_net.forward(new_state_batch).max(dim=1)[0]
        # if and index of the batch is done (True), then set next reward to 0
        target_net[terminal_batch] = 0.0
        q_target = reward_batch + self.gamma * target_net
        loss = self.policy_net.loss(q_target, policy_net).to(self.policy_net.device)
        loss.backward()
        self.policy_net.optimizer.step()
        self.learn_step_counter += 1

        if self.epsilon > self.eps_end:
            self.epsilon -= self.eps_dec
        else:
            self.epsilon = self.eps_end
