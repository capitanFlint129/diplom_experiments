import torch
from absl import app
from compiler_gym.envs import LlvmEnv
from compiler_gym.leaderboard.llvm_instcount import eval_llvm_instcount_policy
from agent import Agent

from main import config, get_observation


def rollout(agent, env, config):
    env.reset()
    observation = get_observation(
        [env.observation[space_name] for space_name in config["observation_spaces"]]
    )
    action_seq, rewards = [], []
    agent.actions_taken = []
    change_count = 0

    for i in range(config["episode_length"]):
        action = agent.choose_action(observation)
        flag = config["actions"][action]
        action_seq.append(action)
        observation, reward, done, info = env.step(
            env.action_space.flags.index(flag),
            observation_spaces=config["observation_spaces"],
        )
        observation = get_observation(observation)
        rewards.append(reward)

        if reward == 0:
            change_count += 1
        else:
            change_count = 0

        if done or change_count > config["patience"]:
            break

    return sum(rewards)


def run(env: LlvmEnv) -> None:
    n_actions = len(config["actions"])
    observation = get_observation(
        [env.observation[space_name] for space_name in config["observation_spaces"]]
    )
    input_dims = observation.shape
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    agent = Agent(
        input_dims=input_dims, n_actions=n_actions, config=config, device=device
    )
    # todo загружать последний run
    agent.policy_net.load_state_dict(torch.load("./models/gallant-valley-71.pth"))
    rollout(agent, env, config)


if __name__ == "__main__":
    app.run(eval_llvm_instcount_policy(run))
