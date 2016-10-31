import numpy as np 
import gym

# Analysis
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd 
import toolz

# Logging
import logging
# get a logger, set the logging level
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# The function itself
import varcompfa as vcf 


# Miscellaneous utility functions
def unit(length, ix):
    """A unit vector of size `length` with index `ix` set to 1, other entries 0."""
    ret = np.zeros(length)
    ret[ix] = 1
    return ret

def bitvector(length, *ixs):
    """A binary vector of size `length` with indices `ixs` set to 1, other entries 0."""
    ret = np.zeros(length)
    ret[[ixs]] = 1
    return ret

def window_avg(seq, n=5):
    """Compute the average within a window of size `n` for each entry of `seq`."""
    kernel = np.ones(n)/n
    return np.convolve(seq, kernel, mode='valid')

def subsample(seq, n):
    """Subsample the array-like `seq`, return `n` evenly spaced values from it."""
    arr = np.array(seq)
    indices = np.rint((len(arr)-1)*np.linspace(0, 1, n)).astype(int)
    return arr[indices]

def log_dict(dct, level=logging.INFO, logger=logging.getLogger()):
    """Log dictionaries with nicely aligned columns."""
    longest = max([len(str(x)) for x in dct.keys()])
    for key in sorted(dct.keys(), key=lambda x: str(x)):
        format_string = '{0:%d}: {1}'%longest
        logger.log(level, format_string.format(key, dct[key]))

def print_dict(dct):
    """Print dictionaries with nicely aligned columns. 
    NB: depending on use-case, you might be better off using JSON instead.
    """
    longest = max([len(str(x)) for x in dct.keys()])
    for key in sorted(dct.keys(), key=lambda x: str(x)):
        format_string = '{0:%d}: {1}'%longest
        print(format_string.format(key, dct[key]))


# Testing linear-Q learning
if True and __name__ == "__main__":    
    # Specify experiment
    num_episodes = 5000
    max_steps = 2000

    # Set up the experiment
    env = gym.make('MountainCar-v0')
    tiling = vcf.UniformTiling(env.observation_space, 20)
    phi = vcf.BinaryVector(tiling.high, tiling)

    # Set up agent
    nf = phi.size
    na = env.action_space.n
    agent = vcf.DiscreteQ(nf, na, epsilon=0.05)
    alpha = 0.01
    gamma = 0.5
    lmbda = 0.0

    # Optimistic Q-value initialization
    agent.w += 1

    # Initialize values
    reward = 0
    done = False

    # Setup tracking
    errlst = []
    rwdlst = []
    history = []
    stepcount = []
    # Run the experiment
    for i in range(num_episodes):
        logger.info("Starting episode: %d"%i)
        obs = env.reset()
        x = phi(obs)
        for j in range(max_steps):
            action = agent.act(x)
            obs, reward, done, _ = env.step(action)

            # compute next state features
            xp = phi(obs)
            
            # update learning algorithm
            delta = agent.learn(x, action, reward, xp, alpha, gamma, lmbda)

            # log information about the timestep
            errlst.append(delta)
            history.append(dict(
                obs=obs.copy(),
                action=action, 
                reward=reward, 
                done=done,
                fvec=x.copy(),
                delta=delta,
            ))

            # Debugging
            # if abs(delta) > 10:
            #     logger.debug("High correction!")
            #     logger.debug(tiling(obs))
            #     # logger.debug(agent.w)
            #     logger.debug(agent.get_value(x, action))
            #     logger.debug(agent.get_value(xp))
            if np.max(np.abs(agent.w)) > 10000:
                logger.debug("Divergence")
                break

            # exit if done, otherwise set up for next iteration
            if done:
                logger.info("Reached terminal state in %d steps"%j)
                break
            x = xp

        else:
            logger.info("Failed to reach the end before the time limit")

        # log end of episode information
        stepcount.append(j)

    # plot the results of the run
    # errors over time
    fig, ax = plt.subplots()
    errors = subsample(window_avg(errlst, 10), 300)
    # rewards = subsample(window_avg(rwdlst, 10), 300)
    ax.plot(errors)
    # ax.plot(rewards)
    plt.show()

    # # heatmap of values
    fig, ax = plt.subplots()
    # get the ranges for the observation_space (assuming 2D)
    space = env.observation_space
    xlim, ylim = list(zip(space.low, space.high))
    xx = np.linspace(*xlim, endpoint=False)
    yy = np.linspace(*ylim, endpoint=False)

    # grid the data points
    shape = (len(xx), len(yy))
    indices = np.ndindex(shape)
    grid = np.array([(xx[i], yy[j]) for i, j in indices])
    data = np.array([phi(i) for i in grid])
    Z = np.max(np.dot(agent.w, data.T), axis=0)
    Z = Z.reshape(*shape)

    # Heatmap with colorbar
    cax = ax.imshow(Z, aspect='equal', interpolation='nearest', cmap=cm.coolwarm)
    ax.set_title('Greedy state-values')
    cbar = fig.colorbar(cax)
    plt.show()
