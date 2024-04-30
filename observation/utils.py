import multiprocessing
import queue
import subprocess
import tempfile
from typing import Optional

import ir2vec
import numpy as np

from config.config import TrainConfig, RAW_IR_OBSERVATION_NAME, LLC_BIN
from env.my_env import MyEnv


def get_observation(env, config: TrainConfig) -> np.ndarray:
    observations = []
    are_all_observations_correct = True
    for observation_name in config.observation_space:
        observation, is_observation_correct = _get_one_observation(
            env, observation_name
        )
        are_all_observations_correct = (
            are_all_observations_correct and is_observation_correct
        )
        observations.append(observation)
    observation = np.concatenate(observations)
    return observation


class ObservationModifier:
    def __init__(self, env, modifications: list[str], episode_length: int):
        self._modifications = modifications
        self._episode_length = episode_length
        self._base_observations_history = []
        self._start_observation = None

        start_observations = []
        for modification in modifications:
            if modification.startswith("start"):
                observation_name = modification.split("-")[1]
                observation, _ = _get_one_observation(env, observation_name)
                start_observations.append(observation)
        if len(start_observations) > 0:
            self._start_observation = np.concatenate(start_observations)

    def modify(
        self,
        base_observation: np.ndarray,
        remains: int,
        env: Optional[MyEnv] = None,
        ir: Optional[str] = None,
    ) -> np.ndarray:
        observation = base_observation.copy()
        self._base_observations_history.append(base_observation.copy())
        if self._start_observation is not None:
            observation = np.concatenate((observation, self._start_observation))
        for modifier in self._modifications:
            if modifier.startswith("remains-counter"):
                counter = remains
                if modifier == "remains-counter-normalized":
                    counter /= self._episode_length
                observation = np.concatenate((observation, np.array([counter])))
            elif modifier.startswith("prev"):
                prev_n = int(modifier.split("-")[1])
                prev = []
                for i in range(1, prev_n):
                    index = max(len(self._base_observations_history) - i - 1, 0)
                    prev.append(self._base_observations_history[index])
                observation = np.concatenate(prev + [observation])
            elif modifier == "mca":
                if ir is not None:
                    observation = np.concatenate([get_mca_vec_ir(ir), observation])
                elif env is not None:
                    observation = np.concatenate(
                        [get_mca_vec_ir(env.get_cur_ir()), observation]
                    )
                else:
                    raise Exception(
                        "You need to specidy ir or env for mca observation modification"
                    )

        return observation


def parse_mca_vec(mca_output) -> np.ndarray:
    uops_per_cycle_line = mca_output.split("\n", maxsplit=9)[7]
    ipc_line = mca_output.split("\n", maxsplit=9)[8]
    block_rthroughput_line = mca_output.split("\n", maxsplit=9)[8]
    return np.array(
        [
            float(uops_per_cycle_line.split()[-1]),
            float(ipc_line.split()[-1]),
            float(block_rthroughput_line.split()[-1]),
        ]
    )


def get_mca_result_from_ir(bc_path):
    proc = subprocess.run(
        f"{LLC_BIN} {bc_path} -o - | llvm-mca",
        capture_output=True,
        shell=True,
        encoding="utf-8",
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def get_mca_result_from_ir_str(ir: str):
    proc = subprocess.run(
        f"{LLC_BIN} -o - | llvm-mca",
        input=ir,
        capture_output=True,
        shell=True,
        encoding="utf-8",
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def parse_rblock_throughput(mca_output):
    line = mca_output.split("\n", maxsplit=9)[8]
    # print(line)
    return float(line.split()[2])


def get_rblock_throughput_bc(bc_path):
    return parse_rblock_throughput(get_mca_result_from_ir(bc_path))


def get_rblock_throughput_ir(ir):
    return parse_rblock_throughput(get_mca_result_from_ir_str(ir))


def get_mca_vec_ir(ir: str) -> np.ndarray:
    return parse_mca_vec(get_mca_result_from_ir_str(ir))


def _get_one_observation(env, observation_name: str) -> tuple[np.ndarray, bool]:
    if observation_name == "AutophaseNorm":
        observation = env.observation["Autophase"]
        observation = observation / observation[51]
        is_count_observation_correct = _is_count_observation_correct(observation)
    elif observation_name == "IR2Vec" or observation_name == "IR2VecNormalized":
        observation = np.array([])
        is_count_observation_correct = False
        for i in range(5):
            result_queue = multiprocessing.Queue()
            ir_text = str(env.observation[RAW_IR_OBSERVATION_NAME])
            proc = multiprocessing.Process(
                target=_get_ir2vec_observation, args=(ir_text, result_queue)
            )
            proc.start()
            proc.join()
            try:
                observation = result_queue.get(timeout=300)
            except queue.Empty:
                print("IR2Vec timeout")
                continue
            if observation_name == "IR2VecNormalized":
                observation = observation / np.linalg.norm(observation)
            is_count_observation_correct = proc.exitcode == 0
            break
    elif observation_name == "InstCount" or observation_name == "InstCountNorm":
        observation = env.observation[observation_name]
        is_count_observation_correct = _is_count_observation_correct(observation)
    else:
        observation = env.observation[observation_name]
        is_count_observation_correct = True
    return observation, is_count_observation_correct


def _get_ir2vec_observation(ir_text: str, result_queue: multiprocessing.Queue) -> None:
    with tempfile.NamedTemporaryFile("w") as ll_file:
        ll_file.write(ir_text)
        ll_file.flush()
        init_obj = ir2vec.initEmbedding(ll_file.name, "fa", "p")
        observation = ir2vec.getProgramVector(init_obj)
    observation = np.array(observation)
    result_queue.put(observation)


def _is_count_observation_correct(observation):
    # в некоторых датасетах встречаются странные программы,
    # дающие нулевые вектора для observation в которых используется
    # подсчет инструкций, что явно является ошибкой если программа не пустая
    # возможно это баг compiler gym
    return (observation > 0).sum() != 0 and not np.any(np.isnan(observation))
