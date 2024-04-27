from config.action_config import O23_SUBSEQ_POSET_LIKE
from config.config import TrainConfig
from utils import make_env

config = TrainConfig()
env = make_env(config)

result = []

for seq in O23_SUBSEQ_POSET_LIKE:
    cur_seq = []
    for action in seq.split():
        if action in env.action_space.flags:
            cur_seq.append(action)
    result.append(cur_seq)

print(result)
