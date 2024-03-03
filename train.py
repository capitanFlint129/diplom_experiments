from dataclasses import asdict

# noinspection PyUnresolvedReferences
import compiler_gym
import torch

import wandb
from config import TrainConfig
from dqn.dqn import Agent
from dqn.train import train, validate
from utils import prepare_datasets, make_env, fix_seed, MODELS_DIR


def main():
    config = TrainConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = wandb.init(
        project="rl-compilers-experiments",
        config=asdict(config),
        # mode="disabled",
    )
    with make_env(config) as train_env:
        fix_seed(config.random_state)
        train_benchmarks, val_benchmarks, test_benchmarks = prepare_datasets(
            train_env,
            config.datasets,
            train_val_test_split=config.train_val_test_split,
            skipped=set(config.skipped_benchmarks),
        )
        agent = Agent(
            observation_size=config.observation_size,
            n_actions=len(config.actions),
            config=config,
            device=device,
        )
        train(
            run,
            agent,
            train_env,
            config,
            train_benchmarks,
            val_benchmarks,
            # enable_validation=False,
            enable_validation_logs=True,
        )

    # final test
    with make_env(config) as test_env:
        agent = Agent(
            observation_size=config.observation_size,
            n_actions=len(config.actions),
            config=config,
            device=device,
        )
        agent.Q_eval.load_state_dict(torch.load(f"{MODELS_DIR}/{run.name}.pth"))
        agent.eval()
        with torch.no_grad():
            test_result = validate(agent, test_env, config, test_benchmarks)
        print(f"Test geomean: {test_result.geomean_reward}")
        run.summary["test_geomean_reward"] = test_result.geomean_reward
        run.summary["test_mean_walltime"] = test_result.mean_walltime
        for (
            dataset_name,
            geomean_reward,
        ) in test_result.geomean_reward_per_dataset.items():
            run.summary[f"test_geomean_reward_{dataset_name}"] = geomean_reward


if __name__ == "__main__":
    main()
