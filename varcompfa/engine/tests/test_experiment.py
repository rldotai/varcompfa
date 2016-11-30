"""
Tests for experiment running code in `experiment.py`.
"""
import gym
import numpy as np
import varcompfa as vcf


class _CheckCallback(vcf.callbacks.Callback):
    """A callback that checks that everything gets called with the expected
    parameters by the experiment runner.
    """
    def __init__(self):
        # Check how many times the callback was called
        self.experiment_begin = 0
        self.experiment_end = 0
        self.episode_begin = 0
        self.episode_end = 0
        self.step_begin = 0
        self.step_end = 0

    def on_experiment_begin(self, info):
        self.experiment_begin += 1

    def on_experiment_end(self, info):
        self.experiment_end += 1

    def on_episode_begin(self, episode_ix, info):
        self.episode_begin += 1

    def on_episode_end(self, episode_ix, info):
        self.episode_end += 1

    def on_step_begin(self, step_ix, info):
        self.step_begin += 1

    def on_step_end(self, step_ix, info):
        self.step_end += 1



class TestPolicyEvaluation:
    def test_init(self):
        env = gym.make('MountainCar-v0')
        pol = vcf.DiscreteRandomControl(env.action_space.n)
        agents = []
        experiment = vcf.PolicyEvaluation(env, pol, agents)

    def test_run(self):
        # Set up environment and policy
        env = gym.make('MountainCar-v0')
        pol = vcf.DiscreteRandomControl(env.action_space.n)

        # Define an agent
        phi_1 = vcf.BiasUnit()
        td_1 = vcf.TD(len(phi_1))
        params_1 = {
            'alpha' : 0.01,
            'gm': 0.999,
            'gm_p': 0.999,
            'lm': 0.1,
        }
        agent_1 = vcf.Agent(td_1, phi_1, params_1)
        agents = [agent_1]

        # Set up the experiment
        experiment = vcf.PolicyEvaluation(env, pol, agents)

        # Try running the experiment
        num_eps = 10
        max_steps = 10

        experiment.run(num_eps, max_steps, callbacks=[])

    def test_run_with_callbacks(self):
        # Set up environment and policy
        env = gym.make('MountainCar-v0')
        pol = vcf.DiscreteRandomControl(env.action_space.n)

        # Define an agent
        phi_1 = vcf.BiasUnit()
        td_1 = vcf.TD(len(phi_1))
        params_1 = {
            'alpha' : 0.01,
            'gm': 0.999,
            'gm_p': 0.999,
            'lm': 0.1,
        }
        agent_1 = vcf.Agent(td_1, phi_1, params_1)
        agents = [agent_1]

        # Set up the experiment
        experiment = vcf.PolicyEvaluation(env, pol, agents)

        # Set up testing callbacks
        cbk = _CheckCallback()

        # Try running the experiment
        num_eps = 10
        max_steps = 10
        experiment.run(num_eps, max_steps, callbacks=[cbk])

        # Check that the callbacks ran properly
        assert(cbk.experiment_begin > 0)
        assert(cbk.experiment_end > 0)
        assert(cbk.episode_begin > 0)
        assert(cbk.episode_end > 0)
        assert(cbk.step_begin > 0)
        assert(cbk.step_end > 0)
