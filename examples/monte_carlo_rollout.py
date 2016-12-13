"""
Example of Monte-Carlo rollouts starting from different initial states.
Here, we use a grid search method, although it would be possible to start from
randomly sampled points as well.
"""
import itertools
import json_tricks as jt
import numpy as np
import pandas as pd
import toolz
import varcompfa as vcf

# Plotting (Move elsewhere?)
import matplotlib as mpl
import matplotlib.pyplot as plt

# Logging
import logging
logger = logging.getLogger(__name__)


CONTROL_PATH = '../data/mountain_car_control_agent.json'
# CONTROL_PATH = './mountain_car_control_agent.json'


def map2d(xdata, ydata, func):
    """Given two sequences `xdata` and `ydata`, and a function `func`, return a
    2D-array `grid` whose i-jth entry is `func((xdata[i], ydata[j]))`.

    NOTE:
    -----
    We pass the value to `func` as a single argument.
    """
    xdata = np.squeeze(xdata)
    ydata = np.squeeze(ydata)
    assert(xdata.ndim == ydata.ndim == 1)
    nx = len(xdata)
    ny = len(ydata)
    indices = np.ndindex((nx, ny))
    # Appy function to data and reshape array
    grid = np.reshape([func((xdata[i], ydata[j])) for i, j in indices], (nx, ny))
    return grid

def grid_space(space, n=50):
    """Return a sequence of linear interval arrays for a given `space` (of the
    kind provided by OpenAI gym.
    """
    assert(isinstance(space, gym.core.Space))
    return [np.linspace(lo, hi, num=n) for lo, hi in zip(space.low, space.high)]

def my_heatmap(data, **kwargs):
    """Custom heatmap featuring a colorbar"""
    fig, ax = plt.subplots()
    cax = ax.imshow(data, aspect='equal', interpolation='nearest', cmap=cm.coolwarm)
    cbar = fig.colorbar(cax)
    return fig, ax, cax, cbar



if __name__ == "__main__":
    import gym
    # Create the environment
    env = gym.make('MountainCar-v0')

    # Load the control agent
    control = jt.load(open(CONTROL_PATH, 'r'))

    # Define a learning agent
    # Tile coding for discretization to binary vectors
    tiling_1    = vcf.features.BinaryTiling(env.observation_space, 17)
    tiling_2    = vcf.features.BinaryTiling(env.observation_space, 19)
    bias        = vcf.features.BiasUnit()
    # Concatenate binary vectors
    phi         = vcf.Union(bias, tiling_1, tiling_2)

    # Parameters for the agent
    td_params = {
        'alpha' : vcf.parameters.EpisodicPowerLaw(0.15, 0.5),
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
        hist_cbk,
    ]

    # Initialize via grid-search
    space = env.observation_space
    gamuts = [np.linspace(lo, hi, num=50)[1:-1] for lo, hi in zip(space.low, space.high)]
    # Get pairs of starting locations
    grid = np.array([np.ravel(i) for i in np.meshgrid(*gamuts)])
    # Make an infinite iterator
    grid_starts = itertools.cycle(grid.T)

    # Random initialization
    # random_starts = (env.observation_space.sample() for i in itertools.count())

    # Select initialization
    # init_states = toolz.interleave([grid_starts, random_starts])
    init_states = grid_starts

    # Run the experiment
    experiment.run(len(grid.T)*10, 2000, callbacks=cbk_lst, initial_states=init_states)


    # # Convert to dataframe
    contexts = hist_cbk.contexts
    df = pd.DataFrame(contexts)
    print(df.head()) # TODO: REMOVE?

    # # Plot values
    values = map2d(*grid_space(env.observation_space), agent.get_value)
    my_heatmap(values)

    plt.show()
