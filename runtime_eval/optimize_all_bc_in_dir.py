import os
import subprocess
import sys
import tempfile
from subprocess import TimeoutExpired

import compiler_gym
import torch
from compiler_gym.envs import CompilerEnv

from config.config import TrainConfig
from observation.utils import ObservationModifier
from utils import (
    get_agent,
    get_model_path,
)
from utils import get_ir2vec_from_file

MODEL_ITERS = 25
RUN_NAME = "misunderstood-sunset-112"


def optimize_file(filepath, config, agent, iters, print_debug=True):
    agent.episode_reset()
    flags = []
    with tempfile.NamedTemporaryFile("w") as ll_file:
        subprocess.run(
            f"llvm-dis {filepath} -o {ll_file.name}", shell=True, capture_output=True
        )
        observation_modifier = ObservationModifier(
            None, config.observation_modifiers, config.episode_length
        )
        for i in range(iters):
            base_observation = get_ir2vec_from_file(ll_file.name)
            observation = observation_modifier.modify(
                base_observation, config.episode_length - i
            )
            action, value = agent.choose_action(
                observation=observation,
                enable_epsilon_greedy=False,
                forbidden_actions=set(),
                eval_mode=True,
            )
            if print_debug:
                print(f"{config.actions[action]}({value})", end=" ")
            flags.append(config.actions[action])
            sequence = config.actions[action]
            if sequence == "noop":
                break
            proc = subprocess.run(
                [
                    "opt",
                    " ".join(sequence),
                    "-S",
                    "-o",
                    ll_file.name,
                    ll_file.name,
                ],
                capture_output=True,
            )
            if proc.returncode != 0:
                print(proc.stderr)
                raise Exception(f"opt: Compilation failed {proc.stderr}")
        proc = subprocess.run(
            ["llvm-as", ll_file.name, "–o", file_path],
            capture_output=True,
        )
        if proc.returncode != 0:
            print(proc.stderr)
            raise Exception(f"opt: Compilation failed {proc.stderr}")
    if print_debug:
        print()
    return flags


if __name__ == "__main__":
    env: CompilerEnv = compiler_gym.make("llvm-v0")

    config = TrainConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = get_agent(
        config,
        device,
        policy_net_path=get_model_path(RUN_NAME),
    )

    bc_dir = sys.argv[2]
    for root, _, files in os.walk(bc_dir):
        for filename in files:
            file_path = os.path.join(root, filename)
            try:
                optimize_file(file_path, config, agent, iters=MODEL_ITERS)
            except TimeoutExpired as e:
                print(f"IR2vec timeout skip benchmark: {e}")
