import datetime
# import json
import os
import time
from pprint import pprint
import numpy as np
import json_tricks

import varcompfa as vcf

# Logging
import logging
# get a logger, set the logging level
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)



class AnnoyinglyVerboseCallback(vcf.callbacks.Callback):
    """An example callback that pretty-prints all information it has access to
    at each point when it gets called.

    It will likely print a lot.
    """
    def on_experiment_begin(self, info=dict()):
        print("Started training")
        # pprint(info)
        print(info.keys())

    def on_experiment_end(self, info=dict()):
        print("End of training")
        # pprint(info)
        print(info.keys())

    def on_episode_begin(self, episode_ix, info=dict()):
        print("Started episode: %d"%episode_ix)
        # pprint(info)
        print(info.keys())

    def on_episode_end(self, episode_ix, info=dict()):
        print("End of episode: %d"%episode_ix)
        # pprint(info)
        print(info.keys())

    def on_step_begin(self, step_ix, info=dict()):
        print("Begin step: %d"%step_ix)
        # pprint(info)
        print(info.keys())

    def on_step_end(self, step_ix, info=dict()):
        print("End step: %d"%step_ix)
        # pprint(info)
        print(info.keys())


class History(vcf.callbacks.Callback):
    def __init__(self):
        # Create data structure that will be filled by running the experiment
        self.hist = {}

    def on_experiment_begin(self, info=dict()):
        # TODO: Compute some of these in the PolicyEval class?
        self.hist['version'] = vcf.utils.current_version()
        self.hist['git_hash'] = vcf.utils.current_git_hash()
        self.hist['start_time'] = time.time()
        self.hist['episodes'] = list()

    def on_experiment_end(self, info=dict()):
        # TODO: Compute in the PolicyEval class?
        self.hist['end_time'] = time.time()

    def on_episode_begin(self, episode_ix, info=dict()):
        self.episode = {
            'steps': list(),
            'updates': list(),
        }

    def on_episode_end(self, episode_ix, info=dict()):
        self.hist['episodes'].append(self.episode)

    def on_step_begin(self, step_ix, info=dict()):
        pass

    def on_step_end(self, step_ix, info=dict()):
        self.episode['steps'].append(info['context'])
        self.episode['updates'].append(info['update_results'])

    def __str__(self):
        return json_tricks.dumps(self.hist, indent=2)




# An example using the simple MDP
if __name__ == "__main__" and True:
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

    # Define the agents to update
    agents = [control, td_agent1, td_agent2]

    # Set up the experiment
    experiment = vcf.PolicyEvaluation(env, control, agents=agents)

    # Run the experiment
    # experiment.run(100, 10, [AnnoyinglyVerboseCallback()])
    hist_cbk = History()
    experiment.run(10, 10, [hist_cbk])

    print('*'*80)
    # print(hist_cbk)


# An example using the MountainCar domain
# Currently, the history of the run is very large, and recording it requires
# a substantial amount of memory
if __name__ == "__main__" and False:
    import gym
    env = gym.make('MountainCar-v0')
    na = env.action_space.n

    # Define components of feature function
    tile_1 = vcf.UniformTiling(env.observation_space, 5)
    tile_2 = vcf.UniformTiling(env.observation_space, 19)
    bvec_1 = vcf.BinaryVector(tile_1.high, tile_1)
    bvec_2 = vcf.BinaryVector(tile_2.high, tile_2)
    # Define the feature function
    phi = vcf.Union(vcf.BiasUnit(), bvec_1, bvec_2)

    # Define the control (discrete actions Q-learning)
    dq = vcf.DiscreteQ(len(phi), na, epsilon=0.02)
    dq_params = {
        'alpha' : vcf.Constant(0.01),
        'gm'    : vcf.Constant(0.9999, 0),
        'gm_p'  : vcf.Constant(0.9999, 0),
        'lm'    : vcf.Constant(0.1, 0),
    }
    control = vcf.Agent(dq, phi, dq_params)

    # List of agents to update
    agents = [control]

    # Set up the experiment
    experiment = vcf.PolicyEvaluation(env, control, agents=agents)

    # Run the experiment
    # experiment.run(100, 10, [AnnoyinglyVerboseCallback()])
    hist_cbk = History()
    experiment.run(10, 5000, [hist_cbk])

    print('*'*80)
    # print(hist_cbk)
    print(len(str(hist_cbk)))
    print(len(json_tricks.dumps(hist_cbk.hist, compression=True)))
