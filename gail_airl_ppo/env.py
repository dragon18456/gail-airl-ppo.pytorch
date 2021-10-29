import gym
import dmc2gym

gym.logger.set_level(40)


def make_env(env_id):
    return NormalizedEnv(gym.make(env_id))

def make_dmc_env(domain_id, task_id):
    return NormalizedEnv(dmc2gym.make(domain_name=domain_id, task_name=task_id))

class NormalizedEnv(gym.Wrapper):

    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self._max_episode_steps = env._max_episode_steps

        self.scale = env.action_space.high
        self.action_space.high /= self.scale
        self.action_space.low /= self.scale

    def step(self, action):
        return self.env.step(action * self.scale)
