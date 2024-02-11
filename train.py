# noinspection PyUnresolvedReferences
import compiler_gym
import torch

import wandb
from dqn.dqn import Agent
from dqn.train import train, validate
from utils import prepare_datasets, make_env, fix_seed

config = dict(
    # Algorithm section
    algorithm="DQN",
    gamma=0.90,  # The percent of how often the actor stays on policy
    epsilon=1.0,  # The starting value for epsilon
    epsilon_end=0.05,  # The ending value for epsilon
    epsilon_dec=5e-5,  # The decrement value for epsilon
    alpha=0.001,  # The learning rate
    batch_size=32,  # The batch size
    max_mem_size=100000,  # The maximum memory size
    replace=500,  # The number of iterations to run before replacing target network
    fc_dim=128,  # The dimension of a fully connected layer
    episodes=4000,  # The number of episodes used to learn
    episode_length=20,  # The (MAX) number of transformation passes per episode
    patience=4,  # The (MAX) number of times to apply a series of transformations without observable change
    learn_memory_threshold=32,  # The number of fully exploratory episodes to run before starting learning
    # General section
    datasets=[
        "benchmark://cbench-v1",
        "benchmark://mibench-v1",
        "benchmark://opencv-v0",
    ],
    no_split=False,
    compiler_gym_env="llvm-v0",
    observation_space="InstCountNorm",
    observation_space_shape=[69],
    reward_space="IrInstructionCountOz",
    logging_history_size=100,
    actions=[
        "-add-discriminators",
        "-adce",
        "-aggressive-instcombine",
        "-alignment-from-assumptions",
        "-always-inline",
        "-argpromotion",
        "-attributor",
        "-barrier",
        "-bdce",
        "-break-crit-edges",
        "-simplifycfg",
        "-callsite-splitting",
        "-called-value-propagation",
        "-canonicalize-aliases",
        "-consthoist",
        "-constmerge",
        "-constprop",
        "-coro-cleanup",
        "-coro-early",
        "-coro-elide",
        "-coro-split",
        "-correlated-propagation",
        "-cross-dso-cfi",
        "-deadargelim",
        "-dce",
        "-die",
        "-dse",
        "-reg2mem",
        "-div-rem-pairs",
        "-early-cse-memssa",
        "-early-cse",
        "-elim-avail-extern",
        "-ee-instrument",
        "-flattencfg",
        "-float2int",
        "-forceattrs",
        "-inline",
        "-insert-gcov-profiling",
        "-gvn-hoist",
        "-gvn",
        "-globaldce",
        "-globalopt",
        "-globalsplit",
        "-guard-widening",
        "-hotcoldsplit",
        "-ipconstprop",
        "-ipsccp",
        "-indvars",
        "-irce",
        "-infer-address-spaces",
        "-inferattrs",
        "-inject-tli-mappings",
        "-instsimplify",
        "-instcombine",
        "-instnamer",
        "-jump-threading",
        "-lcssa",
        "-licm",
        "-libcalls-shrinkwrap",
        "-load-store-vectorizer",
        "-loop-data-prefetch",
        "-loop-deletion",
        "-loop-distribute",
        "-loop-fusion",
        "-loop-guard-widening",
        "-loop-idiom",
        "-loop-instsimplify",
        "-loop-interchange",
        "-loop-load-elim",
        "-loop-predication",
        "-loop-reroll",
        "-loop-rotate",
        "-loop-simplifycfg",
        "-loop-simplify",
        "-loop-sink",
        "-loop-reduce",
        "-loop-unroll-and-jam",
        "-loop-unroll",
        "-loop-unswitch",
        "-loop-vectorize",
        "-loop-versioning-licm",
        "-loop-versioning",
        "-loweratomic",
        "-lower-constant-intrinsics",
        "-lower-expect",
        "-lower-guard-intrinsic",
        "-lowerinvoke",
        "-lower-matrix-intrinsics",
        "-lowerswitch",
        "-lower-widenable-condition",
        "-memcpyopt",
        "-mergefunc",
        "-mergeicmps",
        "-mldst-motion",
        "-sancov",
        "-name-anon-globals",
        "-nary-reassociate",
        "-newgvn",
        "-pgo-memop-opt",
        "-partial-inliner",
        "-partially-inline-libcalls",
        "-post-inline-ee-instrument",
        "-functionattrs",
        "-mem2reg",
        "-prune-eh",
        "-reassociate",
        "-redundant-dbg-inst-elim",
        "-rpo-functionattrs",
        "-rewrite-statepoints-for-gc",
        "-sccp",
        "-slp-vectorizer",
        "-sroa",
        "-scalarizer",
        "-separate-const-offset-from-gep",
        "-simple-loop-unswitch",
        "-sink",
        "-speculative-execution",
        "-slsr",
        "-strip-dead-prototypes",
        "-strip-debug-declare",
        "-strip-nondebug",
        "-strip",
        "-tailcallelim",
        "-mergereturn",
    ],
    random_state=42,
)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = wandb.init(
        project="rl-compilers-experiments",
        config=config,
    )
    env = make_env(config)
    fix_seed(config["random_state"])
    train_benchmarks, val_benchmarks, test_benchmarks = prepare_datasets(
        env, config["datasets"], no_split=config["no_split"]
    )
    agent = Agent(
        input_dims=config["observation_space_shape"],
        n_actions=len(config["actions"]),
        config=config,
        device=device,
    )
    train(run, agent, env, config, train_benchmarks, val_benchmarks)
    env.close()

    # final test
    env = make_env(config)
    agent = Agent(
        input_dims=config["observation_space_shape"],
        n_actions=len(config["actions"]),
        config=config,
        device=device,
    )
    agent.Q_eval.load_state_dict(torch.load(f"models/{run.name}.pth"))
    agent.eval()
    with torch.no_grad():
        test_result = validate(agent, env, config, test_benchmarks)
    print(f"Test geomean: {test_result.geomean_reward}")
    run.summary["test_geomean_reward"] = test_result.geomean_reward
    run.summary["test_mean_walltime"] = test_result.mean_walltime
    for dataset_name, geomean_reward in test_result.geomean_reward_per_dataset.items():
        run.summary[f"test_geomean_reward_{dataset_name}"] = geomean_reward
    env.close()


if __name__ == "__main__":
    main()
