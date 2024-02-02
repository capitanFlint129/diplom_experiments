import torch
from absl import app
from compiler_gym.envs import LlvmEnv
from compiler_gym.leaderboard.llvm_instcount import eval_llvm_instcount_policy
from compiler_gym.wrappers import (
    ConstrainedCommandline,
)

from config import config
from dqn import DQN
from train import make_env
from wrapper import PatienceWrapper


def run(env: LlvmEnv) -> None:
    env.observation_space = config["observation_space"]
    env = PatienceWrapper(
        ConstrainedCommandline(
            env,
            flags=config["actions"],
        ),
        patience=config["patience"],
    )

    state = env.reset()
    done = False
    flags = []
    reward_sum = 0
    while not done:
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        action = policy_net(state).max(1).indices.view(1, 1)
        flag = config["actions"][action]
        flags.append(flag)
        state, reward, done, info = env.step(env.action_space.flags.index(flag))
        reward_sum += reward
    print(f'{str(env.benchmark)} - reward_sum: {reward_sum} - flags: {" ".join(flags)}')


if __name__ == "__main__":
    BATCH_SIZE = 128
    GAMMA = 0.99
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 1000
    TAU = 0.005
    LR = 1e-4
    config.update(
        {
            "epsilon": 0,
            "patience": 4,
            "observation_space": "InstCountNorm",
        }
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with make_env() as env:
        input_dims = env.observation_space.shape
    policy_net = (
        DQN(
            input_dims=input_dims,
            n_actions=len(config["actions"]),
            fc1_dims=config["fc_dim"],
            fc2_dims=config["fc_dim"],
            fc3_dims=config["fc_dim"],
        )
        .to(device)
        .eval()
    )
    policy_net.load_state_dict(torch.load("models/cerulean-aardvark-50.pth"))

    app.run(eval_llvm_instcount_policy(run))
