#
"""Evaluate deep q network for leaderboard"""
import torch
from absl import app
from compiler_gym.envs import LlvmEnv
from compiler_gym.leaderboard.llvm_instcount import eval_llvm_instcount_policy

from dqn import rollout, Agent


def run(env: LlvmEnv) -> None:
    agent = Agent(n_actions=15, input_dims=[56])
    env.observation_space = "Autophase"
    agent.Q_eval.load_state_dict(torch.load("feasible-planet-57.pth"))
    rollout(agent, env)


if __name__ == "__main__":
    app.run(eval_llvm_instcount_policy(run))
