import pytest
import torch

from dqn.q_value import DQNLSTM
from utils import fix_seed


@pytest.mark.parametrize(
    "seed, batch_size, max_seq_len, n_actions",
    [
        (0, 8, 1, 1),
        (0, 6, 5, 5),
        (0, 42, 20, 20),
        (10, 8, 1, 1),
        (10, 6, 5, 5),
        (10, 42, 20, 20),
        (42, 8, 1, 1),
        (42, 6, 5, 5),
        (42, 42, 20, 20),
    ],
)
def test_forward_and_forward_step_equivalence(seed, batch_size, max_seq_len, n_actions):
    fix_seed(seed)
    model = DQNLSTM(
        observation_size=10, hidden_size=20, fc_dims=30, n_actions=n_actions
    )
    model.eval()

    observation = torch.randn(batch_size, max_seq_len, 10)
    prev_action = torch.randint(0, n_actions, (batch_size, max_seq_len))
    if max_seq_len == 1:
        sequence_length = torch.ones(batch_size, dtype=torch.int64)
    else:
        sequence_length = torch.randint(1, max_seq_len, (batch_size,))

    actions_q_forward = model.forward(observation, prev_action, sequence_length)

    actions_q_step_final = torch.zeros(batch_size, n_actions)
    for i in range(batch_size):
        h_prev, c_prev = None, None
        actions_q_step_list = []
        for t in range(sequence_length[i]):
            observation_t = observation[i, t, :]
            prev_action_t = prev_action[i, t].item()
            actions_q_step, h_prev, c_prev = model.forward_step(
                observation_t, prev_action_t, h_prev, c_prev
            )
            actions_q_step_list.append(actions_q_step)

        actions_q_step_final[i] = actions_q_step_list[-1]

    # Check if the outputs are close enough
    assert torch.allclose(
        actions_q_forward, actions_q_step_final, atol=1e-6
    ), "Outputs from `forward` and `forward_step` are not equivalent"
