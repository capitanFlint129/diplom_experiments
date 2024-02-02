import torch
from absl import app
from compiler_gym.envs import LlvmEnv
from compiler_gym.leaderboard.llvm_instcount import eval_llvm_instcount_policy

from dqn import rollout, Agent
from train import config


def run(env: LlvmEnv) -> None:
    env.reset()
    observation = env.observation[config["observation_space"]]
    input_dims = observation.shape
    agent = Agent(input_dims=input_dims, n_actions=len(config["actions"]))
    env.observation_space = config["observation_space"]
    agent.Q_eval.load_state_dict(torch.load("models/tough-microwave-79.pth"))
    rollout(agent, env)


if __name__ == "__main__":
    app.run(eval_llvm_instcount_policy(run))
