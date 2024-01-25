import torch
from absl import app
from compiler_gym.envs import LlvmEnv
from compiler_gym.leaderboard.llvm_instcount import eval_llvm_instcount_policy

from dqn import rollout, Agent
from train import config


def run(env: LlvmEnv) -> None:
    config.update(
        {
            "epsilon": 0,
            "patience": 4,
            "observation_space": "InstCountNorm",
        }
    )
    env.observation_space = config["observation_space"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    agent = Agent(
        config=config,
        input_dims=env.observation[config["observation_space"]].shape,
        n_actions=len(config["actions"]),
        device=device,
    )
    agent.Q_eval.load_state_dict(torch.load("./models/H10-N4000-INSTCOUNTNORM.pth"))
    rollout(config, agent, env)


if __name__ == "__main__":
    app.run(eval_llvm_instcount_policy(run))
