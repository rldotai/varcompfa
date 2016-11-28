import datetime
import os
import time
import numpy as np

import varcompfa as vcf

# Logging
import logging
# get a logger, set the logging level
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class Constant:
    """A constant parameter, which has the option to return a different value
    if the terminal state has been reached.
    """
    def __init__(self, value, terminal_value=None):
        if terminal_value is None:
            terminal_value = value
        self.value = value
        self.terminal_value = terminal_value

    def __call__(self, context):
        if context['done']:
            return self.terminal_value
        return self.value


class ExampleCallback(vcf.callbacks.Callback):
    def on_train_begin(self):
        print("Started training")

    def on_train_end(self):
        print("End of training")

    def on_episode_begin(self, episode_ix):
        print("Started episode: %d"%episode_ix)

    def on_episode_end(self, episode_ix):
        print("End of episode: %d"%episode_ix)

    def on_step_begin(self, step_ix):
        print("Begin step: %d"%step_ix)

    def on_step_end(self, step_ix):
        print("End step: %d"%step_ix)



if __name__ == "__main__":
    import gym

    env = gym.make('SimpleMDP-v0')
    ns = env.observation_space.n
    na = env.action_space.n

    q_params = {
        'alpha' : Constant(0.01),
        'gm'    : Constant(0.999, 0),
        'gm_p'  : Constant(0.999, 0),
        'lm'    : Constant(0.01, 0),
    }
    q_phi = vcf.BinaryVector(ns)
    q_algo = vcf.DiscreteQ(len(q_phi), na, epsilon=0.05)
    control = vcf.Agent(q_algo, q_phi, q_params)


    # Define some other agents that simply learn the value function
    phi1 = vcf.BinaryVector(ns)
    td_params = {
        'alpha' : Constant(0.01),
        'gm'    : Constant(0.999, 0),
        'gm_p'  : Constant(0.999, 0),
        'lm'    : Constant(0.01, 0),
    }
    td_agent1 = vcf.Agent(vcf.TD(len(phi1)), phi1, td_params)

    phi2 = vcf.BiasUnit()
    td_params2 = {
        'alpha' : Constant(0.01),
        'gm'    : Constant(0.999, 0),
        'gm_p'  : Constant(0.999, 0),
        'lm'    : Constant(0.01, 0),
    }
    td_agent2 = vcf.Agent(vcf.TD(len(phi2)), phi2, td_params2)

    agents = [control, td_agent1, td_agent2]
    experiment = vcf.PolicyEvaluation(env, control, agents=agents)
    experiment.run(100, 10, [ExampleCallback()])
