import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from compiler_gym.util.statistics import arithmetic_mean, geometric_mean
from compiler_gym.util.timer import Timer

# Start implementing ideas from Deep RL Bootcamp series on youtube

"""
- concatentate observation with previous observations (they used 4)
- use huber loss (same from [-1,1], but penalizes less for larger errors)
- use RMSProp instead of grad descent
- add more exploration at the beginning
- could try prioritized experience replay again...
- could also try dueling _dqn to see the effect of the advantage property

"""

MODELS_DIR = "models"


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
        self.softmax = nn.Softmax(dim=-1)
        self.optimizer = optim.Adam(self.parameters(), lr=ALPHA)
        self.loss = nn.SmoothL1Loss()  # try huber loss

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        actions = self.softmax(x)
        return actions


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
            actions = self.Q_eval.forward(state)
            # network seems to choose same action over and over, even with zero reward,
            # trying giving negative reward for choosing same action multiple times
            while torch.argmax(actions).item() in self.actions_taken:
                actions[0][torch.argmax(actions).item()] = 0.0
            action = torch.argmax(actions).item()
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


def process_observation(observation):
    # Autophase
    return observation / observation[51]


def save_model(state_dict, model_name, replace=True):
    if not replace and os.path.exists(f"./{MODELS_DIR}/{model_name}.pth"):
        return
    if not os.path.exists(f"./{MODELS_DIR}"):
        os.makedirs(f"./{MODELS_DIR}")
    torch.save(state_dict, f"./{MODELS_DIR}/{model_name}.pth")


def train(run, agent: Agent, env, config, train_benchmarks, val_benchmarks):
    env.observation_space = config["observation_space"]
    history_size = config["logging_history_size"]
    mem_cntr = 0
    history = np.zeros(history_size)
    best_val_geomean = 0

    for episode_i in range(config["episodes"]):
        agent.Q_eval.train()
        env.reset(benchmark=train_benchmarks[episode_i % len(train_benchmarks)])
        observation = process_observation(env.observation[config["observation_space"]])
        done = False
        total = 0
        actions_taken = 0
        agent.actions_taken = []
        change_count = 0

        losses = []
        chosen_flags = []
        while (
            not done
            and actions_taken < config["episode_length"]
            and change_count < config["patience"]
        ):
            action = agent.choose_action(observation)
            flag = config["actions"][action]
            chosen_flags.append(flag)
            new_observation, reward, done, info = env.step(
                env.action_space.flags.index(flag),
                observation_spaces=[config["observation_space"]],
            )
            new_observation = process_observation(new_observation[0])
            actions_taken += 1
            total += reward

            if reward == 0:
                change_count += 1
            else:
                change_count = 0

            agent.store_transition(action, observation, reward, new_observation, done)
            loss_val = agent.learn()
            if loss_val is not None:
                losses.append(loss_val)
            observation = new_observation

            if len(agent.actions_taken) == len(config["actions"]):
                done = True

        index = mem_cntr % history_size
        history[index] = total
        mem_cntr += 1
        print(f"{episode_i} - {env.benchmark}")
        print(
            "Total: {:.4f}".format(total)
            + " Epsilon: {:.4f}".format(agent.epsilon)
            + f" Average rewards sum: {str(np.mean(history))}"
            + f" Action: {' '.join(chosen_flags)}"
        )
        run.log(
            {
                "average_rewards_sum_last_100": np.mean(history),
                "std_rewards_sum_last_100": np.std(history),
                "average_episode_loss": np.mean(losses or [0]),
                "total_episode_reward": total,
            },
            step=episode_i,
        )
        if episode_i % 500 == 0:
            best_val_geomean = _validation(
                run, episode_i, best_val_geomean, agent, env, config, val_benchmarks
            )

    if (config["episodes"] - 1) % 500 != 0:
        _validation(
            run,
            config["episodes"] - 1,
            best_val_geomean,
            agent,
            env,
            config,
            val_benchmarks,
        )
    save_model(agent.Q_eval.state_dict(), run.name, replace=False)


def _validation(run, episode_i, best_val_geomean, agent, env, config, val_benchmarks):
    val_geomean, _ = validate(agent, env, config, val_benchmarks)
    run.log(
        {"val_geomean_reward": val_geomean},
        step=episode_i,
    )
    if val_geomean > best_val_geomean:
        print(
            f"model updated. val_geomean: {val_geomean} - best_val_geomean: {best_val_geomean}"
        )
        save_model(agent.Q_eval.state_dict(), f"{run.name}")
        return val_geomean
    return best_val_geomean


def validate(agent, env, config, val_benchmarks):
    agent.Q_eval.eval()
    rewards = []
    times = []
    for benchmark in val_benchmarks:
        env.reset(benchmark=benchmark)
        with Timer() as timer:
            reward = rollout(agent, env, config)
        rewards.append(reward)
        times.append(timer.time)
    return geometric_mean(rewards), arithmetic_mean(times)


@torch.no_grad()
def rollout(agent: Agent, env, config):
    observation = process_observation(env.observation[config["observation_space"]])
    action_seq, rewards = [], []
    agent.actions_taken = []
    change_count = 0

    for i in range(config["episode_length"]):
        action = agent.choose_action(observation, disable_epsilon_greedy=True)
        flag = config["actions"][action]
        action_seq.append(action)
        observation, reward, done, info = env.step(
            env.action_space.flags.index(flag),
            observation_spaces=[config["observation_space"]],
        )
        observation = process_observation(observation[0])
        rewards.append(reward)

        if reward == 0:
            change_count += 1
        else:
            change_count = 0

        if len(agent.actions_taken) == len(config["actions"]):
            done = True

        if done or change_count > config["patience"]:
            break

    return sum(rewards)
