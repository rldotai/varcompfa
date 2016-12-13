"""
Run a policy evaluating agent.

A template to build off.
"""
import json_tricks as jt
import numpy as np
import pandas as pd
import varcompfa as vcf

# Logging
import logging
logger = logging.getLogger(__name__)


CONTROL_PATH = './mountain_car_control_agent.json'

if __name__ == "__main__":
    import gym
    # Create the environment
    env = gym.make('MountainCar-v0')

    # Load the control agent
    control = jt.load(open(CONTROL_PATH, 'r'))
    control.algo.epsilon = 0

    # Define a learning agent
    # Tile coding for discretization to binary vectors
    tiling_1    = vcf.features.BinaryTiling(env.observation_space, 17)
    tiling_2    = vcf.features.BinaryTiling(env.observation_space, 19)
    tiling_3    = vcf.features.BinaryTiling(env.observation_space, 43)
    bias        = vcf.features.BiasUnit()
    # Concatenate binary vectors
    phi         = vcf.Union(bias, tiling_1, tiling_2, tiling_3)

    # Parameters for the agent
    td_params = {
        'alpha' : vcf.parameters.EpisodicPowerLaw(0.1, 0.5),
        'gm'    : vcf.Constant(0.999, 0),
        'gm_p'  : vcf.Constant(0.999, 0),
        'lm'    : vcf.Constant(0.1, 0),
    }
    # Specify the algorithm
    algo = vcf.algos.TD(len(phi))
    # Combine into agent
    agent = vcf.Agent(algo, phi, td_params)

    # List of agents to update
    learners = [agent]

    # Set up the experiment
    experiment = vcf.LiveExperiment(env, control, learners=learners)

    # Set up callbacks
    hist_cbk = vcf.callbacks.History()
    cbk_lst = [
        vcf.callbacks.Progress(),
        hist_cbk
    ]
    # Run the experiment
    experiment.run(1000, 2000, callbacks=cbk_lst)
