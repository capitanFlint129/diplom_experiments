from action_config import O2_SUBSEQ_POSET_LIKE
import compiler_gym

env = compiler_gym.make("llvm-v0")

result = []

for seq in O2_SUBSEQ_POSET_LIKE:
    cur_seq = []
    for action in seq.split():
        if action in env.action_space.flags:
            cur_seq.append(action)
    result.append(" ".join(cur_seq))

env.close()
print(result)
