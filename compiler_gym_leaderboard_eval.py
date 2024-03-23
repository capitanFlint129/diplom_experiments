import torch
from absl import app
from compiler_gym.envs import LlvmEnv
from compiler_gym.leaderboard.llvm_instcount import eval_llvm_instcount_policy

from config import TrainConfig
from dqn.train import rollout
from utils import fix_seed, get_agent


def run(env: LlvmEnv) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = get_agent(config, device, policy_net_path="_models/wandering-bee-129.pth")
    rollout(agent, env, config)


if __name__ == "__main__":
    config = TrainConfig()
    fix_seed(config.random_state)
    app.run(eval_llvm_instcount_policy(run))

# Current mean walltime: 0.480s / benchmark.
# Current geomean reward: 1.0353
