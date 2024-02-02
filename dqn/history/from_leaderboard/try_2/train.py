from dqn import train, Agent
import gym
import compiler_gym
from absl import app
import wandb


if __name__ == "__main__":
    run = wandb.init(
        project="rl-compilers-experiments",
    )

    env = gym.make("llvm-ic-v0")
    agent = Agent(input_dims=[56], n_actions=15)
    train(run, agent, env)
