import os
from collections import defaultdict

from compiler_gym.util.statistics import arithmetic_mean, geometric_mean
import pandas as pd

LEADERBOARD_RESULTS_DIR = "./leaderboard_results"
RESULT_FILE = "best_result.csv"


def main():
    best_reward_for_benchmark = defaultdict(lambda: float("-inf"))
    best_algorithm_for_benchmark = {}
    for result_filename in os.listdir(LEADERBOARD_RESULTS_DIR):
        algorithm_name = result_filename.split("_", maxsplit=1)[1].split(".")[0]
        df = pd.read_csv(os.path.join(LEADERBOARD_RESULTS_DIR, result_filename))
        for index, row in df.iterrows():
            benchmark = row["benchmark"].split("/")[-1]
            reward = float(row["reward"])
            if best_reward_for_benchmark[benchmark] < reward:
                best_reward_for_benchmark[benchmark] = reward
                best_algorithm_for_benchmark[benchmark] = algorithm_name

    data = {
        "benchmark": [],
        "best_reward": [],
        "best_algorithm": [],
    }
    for benchmark in best_reward_for_benchmark:
        data["benchmark"].append(benchmark)
        data["best_reward"].append(best_reward_for_benchmark[benchmark])
        data["best_algorithm"].append(best_algorithm_for_benchmark[benchmark])

    best_result = pd.DataFrame(data=data)
    best_result.to_csv(RESULT_FILE)

    print(f'geometric_mean: {geometric_mean(data["best_reward"])}')
    print(f'arithmetic_mean: {arithmetic_mean(data["best_reward"])}')


if __name__ == "__main__":
    main()
