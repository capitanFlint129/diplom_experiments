import gym
import numpy as np
from compiler_gym import CompilerEnv

from config.action_config import O3_SEQ, COMPLETE_ACTION_SET
from config.config import TrainConfig
from env.performance_optimization import (
    LlvmMcaEnv,
    get_mca_result_from_ir_str,
    parse_rblock_throughput,
)

config = TrainConfig()

env: CompilerEnv = gym.make("llvm-autophase-ic-v0")

mca_env = LlvmMcaEnv(config, env)

rs = np.random.RandomState(10)
for i in range(1):
    benchmark = env.datasets["benchmark://jotaibench-v0"].random_benchmark(
        random_state=rs
    )
    env.reset(benchmark)
    mca_env.reset()


rewards = []
print(benchmark)
cg_br_init = parse_rblock_throughput(
    (get_mca_result_from_ir_str(env.observation["Ir"]))
)
cg_br_prev = cg_br_init
for action in O3_SEQ:
    cg_reward = 0
    if action in env.action_space.flags:
        env.step(env.action_space.flags.index(action))
        cg_br_new = parse_rblock_throughput(
            (get_mca_result_from_ir_str(env.observation["Ir"]))
        )
        cg_reward = cg_br_prev - cg_br_new
        cg_br_prev = cg_br_new
    r = mca_env.step(action)
    print(f"{action} - {r} - {cg_reward}")
    rewards.append(r)

cg_br_o3 = parse_rblock_throughput((get_mca_result_from_ir_str(env.observation["Ir"])))
env.close()
print(sum(rewards))
print(f"O0: {mca_env._rblock_throughput_initial} - cg: {cg_br_init}")
print(f"O3: {mca_env._o3_rb} - cg: {cg_br_o3}")
print(f"Result: {mca_env._rblock_throughput_initial - sum(rewards)}")
