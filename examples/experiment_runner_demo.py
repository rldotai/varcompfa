import datetime
import json
import os
import time
import numpy as np
from pprint import pprint

import varcompfa as vcf

# Logging
import logging
# get a logger, set the logging level
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)



class ExampleCallback(vcf.callbacks.Callback):
    """An example callback that pretty-prints all information it has access to
    at each point when it gets called.

    It will likely print a lot.
    """
    def on_experiment_begin(self, info=dict()):
        print("Started training")
        pprint(info)

    def on_experiment_end(self, info=dict()):
        print("End of training")
        pprint(info)

    def on_episode_begin(self, episode_ix, info=dict()):
        print("Started episode: %d"%episode_ix)
        pprint(info)

    def on_episode_end(self, episode_ix, info=dict()):
        print("End of episode: %d"%episode_ix)
        pprint(info)

    def on_step_begin(self, step_ix, info=dict()):
        print("Begin step: %d"%step_ix)
        pprint(info)

    def on_step_end(self, step_ix, info=dict()):
        print("End step: %d"%step_ix)
        pprint(info)



if __name__ == "__main__":
    import gym

    env = gym.make('SimpleMDP-v0')
    ns = env.observation_space.n
    na = env.action_space.n

    q_params = {
        'alpha' : vcf.Constant(0.01),
        'gm'    : vcf.Constant(0.999, 0),
        'gm_p'  : vcf.Constant(0.999, 0),
        'lm'    : vcf.Constant(0.01, 0),
    }
    q_phi = vcf.BinaryVector(ns)
    q_algo = vcf.DiscreteQ(len(q_phi), na, epsilon=0.05)
    control = vcf.Agent(q_algo, q_phi, q_params)


    # Define some other agents that simply learn the value function
    phi1 = vcf.BinaryVector(ns)
    td_params = {
        'alpha' : vcf.Constant(0.01),
        'gm'    : vcf.Constant(0.999, 0),
        'gm_p'  : vcf.Constant(0.999, 0),
        'lm'    : vcf.Constant(0.01, 0),
    }
    td_agent1 = vcf.Agent(vcf.TD(len(phi1)), phi1, td_params)

    phi2 = vcf.BiasUnit()
    td_params2 = {
        'alpha' : 0.01,
        'gm'    : 0.9,
        'gm_p'  : 0.9,
        'lm'    : 0.9,
    }
    td_agent2 = vcf.Agent(vcf.TD(len(phi2)), phi2, td_params2)

    agents = [control, td_agent1, td_agent2]
    experiment = vcf.PolicyEvaluation(env, control, agents=agents)
    experiment.run(100, 10, [ExampleCallback()])
