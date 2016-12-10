import json_tricks as jt
import numpy as np
import pandas as pd
import varcompfa as vcf

# Plotting (Move elsewhere?)
import matplotlib as mpl
import matplotlib.pyplot as plt

# Logging
import logging
logger = logging.getLogger(__name__)


CONTROL_PATH = '../data/mountain_car_control_agent.json'
# CONTROL_PATH = './mountain_car_control_agent.json'


if __name__ == "__main__":
    import gym
    # Create the environment
    env = gym.make('MountainCar-v0')

    # Load the control agent
    control = jt.load(open(CONTROL_PATH, 'r'))
    control.algo.epsilon = 0

    # Define a learning agent
    # Tile coding for discretization to binary vectors
    tiling_1    = vcf.features.BinaryTiling(env.observation_space, 5)
    tiling_2    = vcf.features.BinaryTiling(env.observation_space, 19)
    tiling_3    = vcf.features.BinaryTiling(env.observation_space, 31)
    bias        = vcf.features.BiasUnit()
    # Concatenate binary vectors
    phi         = vcf.Union(bias, tiling_1, tiling_2, tiling_3)

    # Parameters for the agent
    td_params = {
        'alpha' : 0.01,
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
    experiment = vcf.PolicyEvaluation(env, control, learners=learners)

    # Set up callbacks
    hist_cbk = vcf.callbacks.History()
    cbk_lst = [
        vcf.callbacks.Progress(),
        hist_cbk
    ]
    # Run the experiment
    experiment.run(1000, 2000, callbacks=cbk_lst)

    # Record the results
    episodes = hist_cbk.history['episodes']
    deltas = [{'delta': i[0]} for ep in episodes for i in ep['update_results']]
    contexts = [i for ep in episodes for i in ep['contexts']]
    steps = [{**i, **j} for i, j in zip(deltas, contexts)]

    # Convert to dataframe
    df = pd.DataFrame(steps)
    print(df.head())
    print(df.tail())
    print(len(steps))

    df.delta.plot()
    plt.show()
