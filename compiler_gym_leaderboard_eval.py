import os

import torch
from absl import app
from compiler_gym.envs import LlvmEnv
from compiler_gym.leaderboard.llvm_instcount import eval_llvm_instcount_policy

from config import MODELS_DIR, WANDB_PROJECT_NAME, TrainConfig
from dqn.train import rollout
from utils import fix_seed, get_agent, get_last_model_wandb_naming


def run(env: LlvmEnv) -> None:
    rollout(agent, env, config, use_actions_masking=config.eval_with_forbidden_actions)


if __name__ == "__main__":
    config = TrainConfig()
    use_actions_masking = True
    models_dir = os.path.join(MODELS_DIR, WANDB_PROJECT_NAME)
    last_model_filename = get_last_model_wandb_naming(models_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Load model from {last_model_filename}\n")
    
    agent = get_agent(config, device, os.path.join(models_dir, last_model_filename))
    config = TrainConfig()
    fix_seed(config.random_state)
    app.run(eval_llvm_instcount_policy(run))

# Current mean walltime: 0.480s / benchmark.
# Current geomean reward: 1.0353
