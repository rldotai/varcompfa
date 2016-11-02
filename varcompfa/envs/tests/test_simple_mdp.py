import gym
import varcompfa


class TestSimpleMDP:
    def test_setup(self):
        env = gym.make('SimpleMDP-v0')
        assert(isinstance(env, gym.Env))

    def test_interaction(self):
        env = gym.make('SimpleMDP-v0')
        obs_space   = env.observation_space
        act_space   = env.action_space
        rmin, rmax  = env.reward_range

        # check that initialization works properly
        obs = env.reset()
        assert(obs_space.contains(obs))

        # check that we can interact with the environment
        for i in range(100):
            act = act_space.sample()
            obs, reward, done, info = env.step(act)
            # Generic environment class checks
            assert(obs_space.contains(obs))
            assert(rmin <= reward <= rmax)
            # MDP specific
            if done:
                assert(obs in env._terminals)
                obs = env.reset()

    def test_bad_action(self):
        env = gym.make('SimpleMDP-v0')

