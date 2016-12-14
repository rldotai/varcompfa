"""
Example of Monte-Carlo rollouts starting from different initial states.
Here, we use a grid search method, although it would be possible to start from
randomly sampled points as well.
"""
import argparse
import itertools
import os
import json_tricks as jt
import numpy as np
import pandas as pd
import toolz
import gym
import varcompfa as vcf

# Logging
import logging
logger = logging.getLogger(__name__)

# Local configuration file
import config
import helper

# Plotting (Move elsewhere?)
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.interpolate



parser = argparse.ArgumentParser()
parser.add_argument('-c', '--controller', type=str,
    help='Controller (i.e., the policy) to evaluate.')
parser.add_argument('--env', type=str,
    help='OpenAI gym environment to run the controller on.')
parser.add_argument('--output', type=str, default=None,
    help='output results destination')
parser.add_argument('--max_steps', default=1000,
    help='maximum number of steps per-episode.')
parser.add_argument('-n', type=int, default=50,
    help='number of intervals per-dimension to initialize from.')
parser.add_argument('--samples', default=1, type=int,
    help='number of times to sample each initial state')
parser.add_argument('--gamma', default=1.0,
    help='discount factor')
parser.add_argument('--every-visit', dest='every_visit', action='store_true',
    help='Use every visit Monte Carlo (default False).')

# def map2d(xdata, ydata, func):
#     """Given two sequences `xdata` and `ydata`, and a function `func`, return a
#     2D-array `grid` whose i-jth entry is `func((xdata[i], ydata[j]))`.

#     NOTE:
#     -----
#     We pass the value to `func` as a single argument.
#     """
#     xdata = np.squeeze(xdata)
#     ydata = np.squeeze(ydata)
#     assert(xdata.ndim == ydata.ndim == 1)
#     nx = len(xdata)
#     ny = len(ydata)
#     indices = np.ndindex((nx, ny))
#     # Appy function to data and reshape array
#     grid = np.reshape([func((xdata[i], ydata[j])) for i, j in indices], (nx, ny))
#     return grid

# def grid_space(space, n=50):
#     """Return a sequence of linear interval arrays for a given `space` (of the
#     kind provided by OpenAI gym.
#     """
#     assert(isinstance(space, gym.core.Space))
#     return [np.linspace(lo, hi, num=n) for lo, hi in zip(space.low, space.high)]

# def my_heatmap(data, **kwargs):
#     """Custom heatmap featuring a colorbar"""
#     fig, ax = plt.subplots()
#     cax = ax.imshow(data, aspect='equal', interpolation='nearest', cmap=cm.coolwarm)
#     cbar = fig.colorbar(cax)
#     return fig, ax, cax, cbar



if __name__ == "__main__":
    logger.info('hi')
    args = parser.parse_args()
    print(args)

    # Create the environment
    env = gym.make(args.env)

    # Load the control agent
    control = helper.load_controller(args.controller, args.env)
    # TODO: REMOVE
    control.algo.epsilon = 0.1

    # Set up the experiment
    experiment = vcf.LiveExperiment(env, control, learners=[])

    # Set up discount factor
    gamma = vcf.parameters.Constant(args.gamma, 0)

    # Set up callbacks
    hist_cbk = vcf.callbacks.History()
    cbk_lst = [
        vcf.callbacks.Progress(),
        hist_cbk,
    ]

    # Initialize via grid-search
    space = env.observation_space
    gamuts = [np.linspace(lo, hi, num=args.n) for lo, hi in zip(space.low, space.high)]

    # Get pairs of starting locations
    grid = np.array([np.ravel(i) for i in np.meshgrid(*gamuts)])
    # Make an infinite iterator
    grid_starts = itertools.cycle(grid.T)

    # Run the experiment
    experiment.run(len(grid.T)*args.samples, args.max_steps,
        callbacks=cbk_lst, initial_states=grid_starts)

    # Compute gammas from context
    contexts = hist_cbk.contexts
    contexts = [{**ctx, 'gm': gamma(ctx)} for ctx in contexts]

    # Convert to dataframe
    df = pd.DataFrame(contexts)

    # Compute returns
    df['G'] = vcf.analysis.calculate_return(df.r, df.gm)

    # Convert observations (numpy arrays) to hashable type tuples
    df['obs'] = df['obs'].apply(tuple)

    if args.every_visit:
        # Every visit Monte Carlo
        grouped     = df.groupby('obs')
    else:
        # First visit Monte Carlo (for each starting state)
        grouped     = df[df['t'] == 0].groupby('obs')

    values      = grouped.aggregate({'G': np.mean})
    variances   = grouped.aggregate({'G': np.var}).fillna(0)

    # Plotting over a grid via interpolation
    fig, (ax0, ax1) = plt.subplots(2)
    xlim, ylim = zip(space.low, space.high)
    gx, gy = np.meshgrid(*gamuts)

    # Get 3D coordinates for values
    val_pts = np.array([(x, y, z) for (x, y), z in values.reset_index().values])
    xx, yy, z_vals = val_pts.T

    # Interpolate to a grid of data
    g_vals = scipy.interpolate.griddata((xx, yy),
        np.array(z_vals),
        (gx, gy),
        method='cubic')

    # Get 3D coordinates for values
    var_pts = np.array([(x, y, z) for (x, y), z in variances.reset_index().values])
    xx, yy, z_vars = var_pts.T

    g_vars = scipy.interpolate.griddata((xx, yy),
        np.array(z_vars),
        (gx, gy),
        method='cubic')

    # Plot the data
    cax0 = ax0.imshow(g_vals,
        extent=(*xlim, *ylim),
        cmap=cm.coolwarm,
        origin='lower',
        aspect='auto',
        interpolation='none')
    # cbar0 = fig.colorbar(cax0)

    # Plot the variances
    cax1 = ax1.imshow(g_vars,
        extent=(*xlim, *ylim),
        cmap=cm.coolwarm,
        origin='lower',
        aspect='auto',
        interpolation='none')
    # cbar1 = fig.colorbar(cax1)


    plt.show()
