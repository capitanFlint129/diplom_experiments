import argparse
from dataclasses import asdict

# noinspection PyUnresolvedReferences
import compiler_gym
import torch

import wandb
from config.config import TrainConfig, WANDB_PROJECT_NAME
from dqn.train import train, validate
from utils import (
    fix_seed,
    get_agent,
    make_env,
    prepare_datasets,
    get_model_path,
)


def main():
    api = wandb.Api()  # set to your entity and project
    num_id = (
        max(
            [
                int(r.name.split("-")[-1])
                for r in api.runs(f"tekkengod/{WANDB_PROJECT_NAME}")
                if r.name.split("-")[-1].isdecimal()
            ]
        )
        + 1
    )
    run_name = f"{args.run_name}-{num_id}"
    config = TrainConfig()
    # assert config.actions[0] == "noop"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.test:
        run = wandb.init(
            project=WANDB_PROJECT_NAME,
            config=asdict(config),
            name="test",
            mode="disabled",
        )
    else:
        run = wandb.init(
            project=WANDB_PROJECT_NAME,
            config=asdict(config),
            name=run_name,
            # mode="disabled",
        )
    config.save(run.name)
    with make_env(config) as train_env:
        fix_seed(config.random_state)
        train_benchmarks, val_benchmarks, test_benchmarks = prepare_datasets(
            run.name,
            train_env,
            random_state=config.random_state,
            config=config,
        )
        agent = get_agent(config, device, policy_net_path=None)
        train(
            run,
            agent,
            train_env,
            config,
            train_benchmarks,
            val_benchmarks,
            enable_validation=True,
            enable_validation_logs=True,
        )

    # final test
    with make_env(config) as test_env:
        agent = get_agent(
            config,
            device,
            policy_net_path=get_model_path(run.name),
        )
        with torch.no_grad():
            test_result = validate(
                agent,
                test_env,
                config,
                test_benchmarks,
                use_actions_masking=False,
                enable_logs=True,
            )
    print(f"Test geomean: {test_result.geomean_reward}")
    run.summary["test_geomean_reward"] = test_result.geomean_reward
    run.summary["test_mean_reward"] = test_result.mean_reward
    run.summary["test_mean_walltime"] = test_result.mean_walltime


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_name", help="run name")
    parser.add_argument(
        "--test",
        help="test",
        action="store_true",
    )
    args = parser.parse_args()

    main()
