"""
Run a policy-improving algorithm in an environment and preserve the resulting
agent.
"""
import time
import json_tricks as jt
import numpy as np
import pandas as pd

import varcompfa as vcf

# Logging
import logging
logger = logging.getLogger(__name__)



# An example using the MountainCar domain
if __name__ == "__main__" and True:
    import gym
    env = gym.make('MountainCar-v0')
    na = env.action_space.n

    # Tile coding for discretization to binary vectors
    tiling_1    = vcf.features.BinaryTiling(env.observation_space, 5)
    tiling_2    = vcf.features.BinaryTiling(env.observation_space, 19)
    tiling_3    = vcf.features.BinaryTiling(env.observation_space, 31)
    # Concatenate binary vectors
    phi         = vcf.Union(tiling_1, tiling_2, tiling_3)

    # Define the control (discrete actions Q-learning)
    dq = vcf.DiscreteQ(len(phi), na, epsilon=0.02)
    dq_params = {
        'alpha' : 0.01,
        'gm'    : 0.9999,
        'gm_p'  : vcf.Constant(0.9999, 0),
        'lm'    : vcf.Constant(0.1, 0),
    }
    control = vcf.Agent(dq, phi, dq_params)

    # List of agents to update
    learners = [control]

    # Set up the experiment
    experiment = vcf.LiveExperiment(env, control, learners=learners)

    # Set up callbacks
    hist_cbk = vcf.callbacks.History()
    cbk_lst = [
        hist_cbk,
        vcf.callbacks.Progress(),
    ]
    # Run the experiment
    experiment.run(100, 2000, callbacks=cbk_lst)

    # Convert to dataframe
    df = pd.DataFrame(hist_cbk.contexts)
    print(df.head())
