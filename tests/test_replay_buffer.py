import numpy as np
import pytest
import torch

from dqn.replay_buffer import ReplayBuffer, ReplayBufferForLSTM, DQNTrainBatch
from config import TrainConfig


@pytest.fixture
def replay_buffer():
    buffer_size = 100
    observation_size = 4
    buffer = ReplayBuffer(buffer_size, observation_size)
    return buffer


@pytest.fixture
def replay_buffer_lstm():
    buffer_size = 100
    observation_size = 4
    buffer = ReplayBufferForLSTM(buffer_size, observation_size)
    return buffer


def test_replay_buffer_store_transition(replay_buffer):
    action = 1
    state = np.random.rand(4)
    reward = 1.0
    new_state = np.random.rand(4)
    terminal = False
    replay_buffer.store_transition(action, state, reward, new_state, terminal)
    assert replay_buffer.get_ready_data_size() == 1


def test_replay_buffer_get_batch(replay_buffer):
    batch_size = 32
    device = torch.device("cpu")
    for _ in range(50):
        action = np.random.randint(4)
        state = np.random.rand(4)
        reward = np.random.rand()
        new_state = np.random.rand(4)
        terminal = np.random.choice([True, False])
        replay_buffer.store_transition(action, state, reward, new_state, terminal)
    batch = replay_buffer.get_batch(batch_size, device)
    assert isinstance(batch, DQNTrainBatch)
    assert batch.batch_size == batch_size
    assert batch.state_batch.shape == (batch_size, 4)
    assert batch.new_state_batch.shape == (batch_size, 4)
    assert batch.reward_batch.shape == (batch_size,)
    assert batch.terminal_batch.shape == (batch_size,)
    assert batch.action_batch.shape == (batch_size,)


def test_replay_buffer_lstm_store_transition(replay_buffer_lstm):
    prev_action = 0
    action = 1
    state = np.random.rand(4)
    reward = 1.0
    new_state = np.random.rand(4)
    done = False
    replay_buffer_lstm.store_transition(
        prev_action, action, state, reward, new_state, done
    )
    replay_buffer_lstm.episode_done()
    assert replay_buffer_lstm.get_ready_data_size() == 1


def test_replay_buffer_lstm_get_batch(replay_buffer_lstm):
    episode_length = TrainConfig().episode_length
    batch_size = 32
    device = torch.device("cpu")
    for _ in range(50):
        prev_action = np.random.randint(4)
        action = np.random.randint(4)
        state = np.random.rand(4)
        reward = np.random.rand()
        new_state = np.random.rand(4)
        done = np.random.choice([True, False])
        replay_buffer_lstm.store_transition(
            prev_action, action, state, reward, new_state, done
        )
        if done:
            replay_buffer_lstm.episode_done()
            replay_buffer_lstm.episode_reset()
    batch = replay_buffer_lstm.get_batch(batch_size, device)
    assert isinstance(batch, DQNTrainBatch)
    assert batch.batch_size == batch_size
    assert batch.state_batch.shape[0] == batch_size
    assert batch.new_state_batch.shape[0] == batch_size
    assert batch.reward_batch.shape == (batch_size,)
    assert batch.terminal_batch.shape == (batch_size,)
    assert batch.action_batch.shape[0] == batch_size
    assert batch.prev_action_batch.shape[0] == batch_size
    assert batch.lengths.shape == (batch_size,)
    assert batch.final_action_batch.shape == (batch_size,)
    assert torch.all(batch.lengths <= episode_length)


def test_replay_buffer_lstm_episode_reset_and_done(replay_buffer_lstm):
    for _ in range(10):
        prev_action = np.random.randint(4)
        action = np.random.randint(4)
        state = np.random.rand(4)
        reward = np.random.rand()
        new_state = np.random.rand(4)
        done = False
        replay_buffer_lstm.store_transition(
            prev_action, action, state, reward, new_state, done
        )
    replay_buffer_lstm.episode_done()
    assert replay_buffer_lstm.get_ready_data_size() == 10
    replay_buffer_lstm.episode_reset()
    for _ in range(5):
        prev_action = np.random.randint(4)
        action = np.random.randint(4)
        state = np.random.rand(4)
        reward = np.random.rand()
        new_state = np.random.rand(4)
        done = False
        replay_buffer_lstm.store_transition(
            prev_action, action, state, reward, new_state, done
        )
    replay_buffer_lstm.episode_done()
    assert replay_buffer_lstm.get_ready_data_size() == 15


def test_replay_buffer_lstm_episode_reset_and_done_complicated():
    replay_buffer = ReplayBufferForLSTM(10, 4)
    episode_1 = [
        (
            np.random.randint(4),
            np.random.randint(4),
            np.float32(np.random.rand(4)),
            np.float32(np.random.rand()),
            np.float32(np.random.rand(4)),
            False,
        )
        for _ in range(6)
    ]
    episode_2 = [
        (
            np.random.randint(4),
            np.random.randint(4),
            np.float32(np.random.rand(4)),
            np.float32(np.random.rand()),
            np.float32(np.random.rand(4)),
            False,
        )
        for _ in range(7)
    ]
    replay_buffer.episode_reset()
    for prev_action, action, state, reward, new_state, done in episode_1:
        replay_buffer.store_transition(
            prev_action, action, state, reward, new_state, done
        )
        assert replay_buffer.get_ready_data_size() == 0
        assert replay_buffer.get_cur_size() == 0
    replay_buffer.episode_done()
    assert replay_buffer.get_ready_data_size() == len(episode_1)
    assert replay_buffer.get_cur_size() == len(episode_1)

    replay_buffer.episode_reset()
    for prev_action, action, state, reward, new_state, done in episode_2:
        replay_buffer.store_transition(
            prev_action, action, state, reward, new_state, done
        )
        assert replay_buffer.get_ready_data_size() == len(episode_1)
        assert replay_buffer.get_cur_size() == len(episode_1)
    replay_buffer.episode_done()
    assert replay_buffer.get_ready_data_size() == len(episode_1) + len(episode_2)
    assert replay_buffer.get_cur_size() == 10

    episode_1_data_in_buffer = list(reversed(episode_1))
    episode_2_data_in_buffer = list(reversed(episode_2))
    buffer_data = (
        episode_2_data_in_buffer[-3:]
        + episode_1_data_in_buffer[-3:]
        + episode_2_data_in_buffer[:4]
    )
    for i, (prev_action, action, state, reward, new_state, done) in enumerate(
        buffer_data
    ):
        assert np.all(replay_buffer.state_mem[i] == state)
        assert np.all(replay_buffer.new_state_mem[i] == new_state)
        assert replay_buffer.action_mem[i] == action
        assert replay_buffer.prev_action_mem[i] == prev_action
        assert replay_buffer.reward_mem[i] == reward
        assert replay_buffer.terminal_mem[i] == done
