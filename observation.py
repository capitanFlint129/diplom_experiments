import multiprocessing
import tempfile

import ir2vec
import numpy as np

RAW_IR_OBSERVATION_NAME = "Ir"


def get_observation(env, config) -> tuple[np.ndarray, bool]:
    if config["observation_space"] == "AutophaseNorm":
        observation = env.observation[config["observation_space"]]
        observation = observation / observation[51]
        return observation, _is_observation_correct(observation)
    elif config["observation_space"] == "IR2Vec":
        # observation = np.random.rand(300)
        result_queue = multiprocessing.Queue()
        ir_text = str(env.observation[RAW_IR_OBSERVATION_NAME])
        proc = multiprocessing.Process(
            target=_get_ir2vec_observation, args=(ir_text, result_queue)
        )
        proc.start()
        proc.join()
        observation = result_queue.get()
        return observation, _is_observation_correct(observation)
    else:
        observation = env.observation[config["observation_space"]]
        return observation, _is_observation_correct(observation)


def _get_ir2vec_observation(ir_text: str, result_queue: multiprocessing.Queue) -> None:
    with tempfile.NamedTemporaryFile("w") as llfile:
        llfile.write(ir_text)
        llfile.flush()
        init_obj = ir2vec.initEmbedding(llfile.name, "fa", "p")
        observation = ir2vec.getProgramVector(init_obj)
    observation = np.array(observation)
    result_queue.put(observation)
    # return np.array(observation)
    # return np.random.rand(300)


def _is_observation_correct(observation):
    return (observation > 0).sum() != 0 and not np.any(np.isnan(observation))
