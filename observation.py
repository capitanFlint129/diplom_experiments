import multiprocessing
import tempfile

import ir2vec
import numpy as np

from config import TrainConfig

RAW_IR_OBSERVATION_NAME = "Ir"


def get_observation(env, config: TrainConfig) -> tuple[np.ndarray, bool]:
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
    return observation, are_all_observations_correct


def _get_one_observation(env, observation_name: str) -> tuple[np.ndarray, bool]:
    if observation_name == "AutophaseNorm":
        observation = env.observation["Autophase"]
        observation = observation / observation[51]
        return observation, _is_observation_correct(observation)
    elif observation_name == "IR2Vec" or observation_name == "IR2VecNormalized":
        # observation = np.random.rand(300)
        result_queue = multiprocessing.Queue()
        ir_text = str(env.observation[RAW_IR_OBSERVATION_NAME])
        proc = multiprocessing.Process(
            target=_get_ir2vec_observation, args=(ir_text, result_queue)
        )
        proc.start()
        proc.join()
        observation = result_queue.get()
        if observation_name == "IR2VecNormalized":
            # observation = normalize(observation, axis=0)
            observation = observation / np.linalg.norm(observation)
        return observation, _is_observation_correct(observation)

    observation = env.observation[observation_name]
    return observation, _is_observation_correct(observation)


def _get_ir2vec_observation(ir_text: str, result_queue: multiprocessing.Queue) -> None:
    with tempfile.NamedTemporaryFile("w") as ll_file:
        ll_file.write(ir_text)
        ll_file.flush()
        init_obj = ir2vec.initEmbedding(ll_file.name, "fa", "p")
        observation = ir2vec.getProgramVector(init_obj)
    observation = np.array(observation)
    result_queue.put(observation)


def _is_observation_correct(observation):
    return (observation > 0).sum() != 0 and not np.any(np.isnan(observation))
