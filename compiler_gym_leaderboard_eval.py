import torch
from absl import app
from compiler_gym.envs import LlvmEnv
from compiler_gym.leaderboard.llvm_instcount import eval_llvm_instcount_policy

from config import TrainConfig
from dqn.dqn import Agent
from dqn.train import rollout
from utils import fix_seed


def run(env: LlvmEnv) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = Agent(
        observation_size=config.observation_size,
        n_actions=len(config.actions),
        config=config,
        device=device,
    )
    agent.policy_net.load_state_dict(torch.load("_models/wandering-bee-129.pth"))
    rollout(agent, env, config)


if __name__ == "__main__":
    config = TrainConfig()
    fix_seed(config.random_state)
    app.run(eval_llvm_instcount_policy(run))

# Current mean walltime: 0.480s / benchmark.
# Current geomean reward: 1.0353
