import gym


class PatienceWrapper(gym.Wrapper):
    def __init__(self, env, patience=5):
        super().__init__(env)
        self.env = env
        self.patience = patience
        self.reward_counter = 0

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        if reward <= 0:
            self.reward_counter += 1
        else:
            self.reward_counter = 0

        if self.reward_counter > self.patience:
            done = True
        return next_state, reward, done, info
