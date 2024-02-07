import torch
from absl import app
from compiler_gym.envs import LlvmEnv
from compiler_gym.leaderboard.llvm_instcount import eval_llvm_instcount_policy

from dqn import rollout, Agent
from train import config
from utils import fix_seed


def run(env: LlvmEnv) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = Agent(
        input_dims=config["observation_space_shape"],
        n_actions=len(config["actions"]),
        config=config,
        device=device,
    )
    agent.Q_eval.load_state_dict(torch.load("models/wandering-bee-129.pth"))
    rollout(agent, env, config)


if __name__ == "__main__":
    fix_seed(config["random_state"])
    app.run(eval_llvm_instcount_policy(run))

# Current mean walltime: 0.480s / benchmark.
# Current geomean reward: 1.0353
