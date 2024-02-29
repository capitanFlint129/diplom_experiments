import torch

from config import TrainConfig
from dqn.train import Agent, validate
from train import make_env, fix_seed, MODELS_DIR
from utils import prepare_datasets, get_last_model_wandb_naming

if __name__ == "__main__":
    config = TrainConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_env(config)
    fix_seed(config.random_state)
    _, _, test_benchmarks = prepare_datasets(
        env,
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
    last_model_filename = get_last_model_wandb_naming(MODELS_DIR)
    print(f"Load model from {last_model_filename}\n")
    agent.Q_eval.load_state_dict(torch.load(f"{MODELS_DIR}/{last_model_filename}"))
    agent.eval()
    with torch.no_grad():
        test_result = validate(agent, env, config, test_benchmarks, enable_logs=True)
    env.close()
    print(f"Geomean reward: {test_result.geomean_reward}")
    print(f"Mean walltime: {test_result.mean_walltime}")
