import numpy as np 
import gym

# Analysis
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd 
import scipy as sp 
import scipy.linalg
import itertools
import toolz
from toolz import concat, pluck

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

def lpluck(ind, seqs, default='__no_default__'):
    """Like `toolz.pluck`, but it returns a list instead of an iterator."""
    return list(toolz.pluck(ind, seqs, default=default))

def lconcat(seqs):
    """Like `toolz.concat`, but it returns a list instead of an iterator."""
    return list(toolz.concat(seqs))

def apluck(ind, seqs, default='__no_default__'):
    """Like `toolz.pluck`, but it returns an array instead of an iterator."""
    return np.array(list(toolz.pluck(ind, seqs, default=default)))

def aconcat(seqs):
    """Like `toolz.concat`, but it returns a array instead of an iterator."""
    return np.array(list(toolz.concat(seqs)))

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


# def split_episodes(hist):
#     ret = []
#     epi = []
#     for step in history:
#         epi.append(step)
#         if step['done']:
#             ret.append(epi)
#             epi = []
#     # append last episode even if it wasn't properly terminated
#     if not step['done']:
#         ret.append(epi)
#     return ret


def episode_return(epi):
    """Calculate the return from an episode."""
    ret = []
    g = 0
    # work backwards from the end of the episode
    for step in reversed(epi):
        g += step['reward']
        ret.append(g)
        g *= step['gm']
    # inverse of reverse 
    ret.reverse()
    return ret

def calculate_return(rewards, gamma):
    """Calculate return from a list of rewards and gamma, which may be a sequence or a constant."""
    ret = []
    g = 0
    # Allow for gamma to be specified as a sequence or a constant
    if not hasattr(gamma, '__iter__'):
        gamma = itertools.repeat(gamma)
    # Work backwards through the 
    for r, gm in reversed(list(zip(rewards, gamma))):
        g += r 
        ret.append(g)
        g *= gm
    # inverse of reverse
    ret.reverse()
    return np.array(ret)

def episode_feature_variance(epi):
    """Calculate the variance per-feature in the return of a single episode."""
    ga = np.array(episode_return(epi))
    xa = np.array(list(toolz.pluck('x', epi)))
    gx = xa.T * ga
    return np.var(gx, axis=1)

def episode_feature_mean(epi):
    """Calculate the average return per-feature in the return of a single episode."""
    ga = np.array(episode_return(epi))
    xa = np.array(list(toolz.pluck('x', epi)))
    gx = xa.T * ga
    return np.mean(gx, axis=1)

def plot_phase(trajectory, smoothing=5):
    """Plotting the phase space of a trajectory."""
    trajectory = np.apply_along_axis(window_avg, 0, trajectory, n=smoothing)
    _x, _y = trajectory.T
    plt.plot(_x, _y)
    plt.show()

def serialize_feature(feat):
    """Serializing the features and their implicit dependency graph."""
    def expand_children(ff):
        _params = ff.params
        print(_params)
        print()
        if 'children' in _params:
            _params['children'] = [expand_children(i) for i in _params['children'] if i]
        return _params
    return expand_children(feat)


# Testing linear-Q learning
if True and __name__ == "__main__":    
    # Specify experiment
    num_episodes = 1000
    max_steps = 3000

    # Set up the experiment
    env = gym.make('MountainCar-v0')
    space = env.observation_space # TODO: REMOVE WHEN DONE DEBUGGING
    
    # Tile coding for discretization
    tiling_1    = vcf.UniformTiling(env.observation_space, 2)
    tiling_2    = vcf.UniformTiling(env.observation_space, 3)
    tiling_3    = vcf.UniformTiling(env.observation_space, 7)
    tiling_4    = vcf.UniformTiling(env.observation_space, 13)
    # Convert tile indices to binary vector
    bvec_1      = vcf.BinaryVector(tiling_1.high, tiling_1) 
    bvec_2      = vcf.BinaryVector(tiling_2.high, tiling_2)
    bvec_3      = vcf.BinaryVector(tiling_3.high, tiling_3)
    bvec_4      = vcf.BinaryVector(tiling_4.high, tiling_4)
    # Concatenate binary vectors
    phi         = vcf.Union(bvec_1, bvec_2, bvec_3, bvec_4)

    # Set up agent
    nf = len(phi)
    na = env.action_space.n
    agent = vcf.DiscreteQ(nf, na, epsilon=0.05)
    alpha_0 = 0.1
    gamma = 0.999
    lmbda = 0.8

    # Optimistic Q-value initialization
    agent.w += 1

    # Initialize values
    reward = 0
    done = False

    # Setup tracking
    stepcount = []
    episodes = []
    visits = np.zeros(len(phi)) # feature activations/visits
    # Run the experiment
    for i in range(num_episodes):
        logger.info("Starting episode: %d"%i)
        history = []
        obs = env.reset()
        x = phi(obs)
        alpha = alpha_0 / (1 + i**(1/2)) # decrease learning rate
        for j in range(max_steps):
            action = agent.act(x)
            obs, reward, done, _ = env.step(action)

            # compute next state features
            xp = phi(obs)
            
            # update learning algorithm
            delta = agent.learn(x, action, reward, xp, alpha, gamma, lmbda)

            # log information about the timestep
            history.append(dict(
                obs=obs.copy(),
                action=action, 
                reward=reward,
                done=done,
                x=x.copy(),
                delta=delta,
                alpha=alpha,
                gm=gamma if not done else 0,
                lm=lmbda,
            ))
            # record visit 
            visits[x != 0] += 1

            if np.max(np.abs(agent.w)) > 10000:
                logger.debug("Divergence")
                break

            # exit if done, otherwise set up for next iteration
            if done:
                logger.info("Reached terminal state in %d steps"%j)
                # reduce exploration slightly each time goal is reached 
                agent.epsilon *= 0.95
                break
            x = xp

        else:
            logger.info("Failed to reach the end before the time limit")

        # log end of episode information
        episodes.append(history)
        stepcount.append(j)

    # Save the weights
    agent.save_weights('weights')
    ###########################################################################
    ###########################################################################
    ###########################################################################
    dd = apluck('delta', history)
    # gammas  = apluck('gamma', history)
    gg = np.array([gamma**ix for ix, _ in enumerate(dd)])
    gc = np.cumprod(gg)
    # pairwise deltas
    DD = np.triu(np.outer(dd, dd))
    # discounting upper triangular matrix
    GG = np.triu(sp.linalg.circulant(gc).T)
    # discounted deltas
    DG = DD*GG
    cc = np.sum(DG, axis=0) # cumulants
    # vv = 

    ###########################################################################
    # TRUNCATE HISTORY # TODO: REMOVE
    # episodes = episodes[-500:]

    # Calculating variances
    # array of features
    xa = np.vstack([list(toolz.pluck('x', ep)) for ep in episodes])
    # get array of returns
    ga = np.hstack([episode_return(ep) for ep in episodes])
    # returns multiplied by feature vector
    gx = xa.T * ga
    va = np.var(gx, axis=1)
    # va = np.vstack([episode_feature_variance(ep) for ep in episodes])

    # variance heatmap 
    traj = np.array(list(toolz.pluck('obs', episodes[-1])))    
    fig, ax = plt.subplots()
    xlim, ylim = list(zip(traj.min(axis=0), traj.max(axis=0)))
    xx = np.linspace(*xlim)
    yy = np.linspace(*ylim)

    shape = (len(xx), len(yy))
    indices = np.ndindex(shape)
    grid = np.array([(xx[i], yy[j]) for i, j in indices])
    data = np.array([phi(i) for i in grid])
    Z = np.dot(data, va)

    # reshape as an image
    Z = Z.reshape(*shape)
    Z = Z.T

    # Heatmap with colorbar
    cax = ax.imshow(Z, 
        aspect='equal', 
        interpolation='nearest',
        cmap=cm.viridis,
        origin='lower',
        extent=[0, 1, 0, 1])

    # Annotations
    ax.set_title('State Variance Heatmap')
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    
    # Set tick labels (subsampling the x-y values, rounding to two decimals)
    nticks  = 8
    ax.set_xticks(np.linspace(0, 1, nticks))
    ax.set_yticks(np.linspace(0, 1, nticks))
    ax.set_xticklabels(np.round(subsample(xx, nticks), 2))
    ax.set_yticklabels(np.round(subsample(yy, nticks), 2))
    cbar = fig.colorbar(cax)
    fig.savefig('test_variance.png')
    plt.show()

    ###########################################################################
    # Calculating variances
    # array of features
    xa = np.vstack([list(toolz.pluck('x', ep)) for ep in episodes])
    # get array of returns
    ga = np.hstack([episode_return(ep) for ep in episodes])
    # returns multiplied by feature vector
    gx = xa.T * ga
    # per-feature variance (is this right?)
    va = np.var(gx, axis=1)


    # Delta squared for variance
    deltas = np.array(list(toolz.pluck('delta', toolz.concat(episodes))))
    dsqa = deltas**2
    # delta-squared return
    dsqret = calculate_return(dsqa, lpluck('gm', concat(episodes)))
    # multiplying by features
    dsqrx = xa.T * dsqret
    
    # averaging for per-feature delta-squared return (is this right?)
    dsvar_w = np.mean(dsqrx, axis=1)

    # least squares solution (this appears to have some sort of issue)
    # dsvar_w, res, *_ = np.linalg.lstsq(xa, dsqret)


    # make heatmap 
    traj = np.array(list(toolz.pluck('obs', episodes[-1])))    
    fig, ax = plt.subplots()
    xlim, ylim = list(zip(traj.min(axis=0), traj.max(axis=0)))
    xx = np.linspace(*xlim)
    yy = np.linspace(*ylim)

    shape = (len(xx), len(yy))
    indices = np.ndindex(shape)
    grid = np.array([(xx[i], yy[j]) for i, j in indices])
    data = np.array([phi(i) for i in grid])
    
    ###########################################################################
    # per-feature variance
    Z1 = np.dot(data, va)
    Z1 = Z1.reshape(*shape)
    Z1 = Z1.T
    
    # delta squared version
    Z2 = np.dot(data, dsvar_w)
    Z2 = Z2.reshape(*shape)
    Z2 = Z2.T

    # plotting
    fig, axes = plt.subplots(2)

    # empirical variance
    ax = axes[0]
    # Heatmap with colorbar
    cax = ax.imshow(Z1, 
        aspect='equal', 
        interpolation='nearest',
        cmap=cm.viridis,
        origin='lower',
        extent=[0, 1, 0, 1])

    # Annotations
    ax.set_title('State Variance Heatmap')
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')

    # Set tick labels (subsampling the x-y values, rounding to two decimals)
    nticks  = 8
    ax.set_xticks(np.linspace(0, 1, nticks))
    ax.set_yticks(np.linspace(0, 1, nticks))
    ax.set_xticklabels(np.round(subsample(xx, nticks), 2))
    ax.set_yticklabels(np.round(subsample(yy, nticks), 2))
    cbar = fig.colorbar(cax)
    
    # Now, comparing with least squares solution
    ax = axes[1]
    cax = ax.imshow(Z2, 
        aspect='equal', 
        interpolation='nearest',
        cmap=cm.viridis,
        origin='lower',
        extent=[0, 1, 0, 1])

    # Annotations
    ax.set_title('Delta-Squared Heatmap')
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')

    # Set tick labels (subsampling the x-y values, rounding to two decimals)
    nticks  = 8
    ax.set_xticks(np.linspace(0, 1, nticks))
    ax.set_yticks(np.linspace(0, 1, nticks))
    ax.set_xticklabels(np.round(subsample(xx, nticks), 2))
    ax.set_yticklabels(np.round(subsample(yy, nticks), 2))
    cbar = fig.colorbar(cax)

    # Show the figure
    plt.show()
    ###########################################################################


    if False:
        # Get some information from the run 
        traj = np.array(list(toolz.pluck('obs', episodes[-1])))

        # # plot the results of the run
        # # run length 
        fig, ax = plt.subplots()
        ax.plot(stepcount)
        ax.set_title('run length')
        ax.set_xlabel('episode')
        ax.set_ylabel('steps')
        fig.savefig('test_runlength.png')
        plt.show()


        # # errors over time
        fig, ax = plt.subplots()
        deltas = np.array(list(toolz.pluck('delta', toolz.concat(episodes))))
        errors = subsample(window_avg(deltas, 10), 300)
        ax.set_title('td-errors')
        ax.set_xlabel('t')
        ax.set_ylabel('δ')
        ax.plot(errors)
        fig.savefig('test_deltas.png')
        plt.show()


        # # heatmap of values
        fig, ax = plt.subplots()
        # get the ranges for the observation_space (assuming 2D)
        xlim, ylim = list(zip(traj.min(axis=0), traj.max(axis=0)))
        xx = np.linspace(*xlim)
        yy = np.linspace(*ylim)

        # grid the data points
        shape = (len(xx), len(yy))
        indices = np.ndindex(shape)
        grid = np.array([(xx[i], yy[j]) for i, j in indices])
        data = np.array([phi(i) for i in grid])
        Z = agent.get_value(data.T)
        
        # zero out non-visited features
        # unseen_value = np.max(np.dot(agent.w[:,visits==0], data[:,visits==0].T), axis=0)
        # Z = Z - unseen_value
        
        # reshape as an image
        Z = Z.reshape(*shape)
        Z = Z.T

        # Heatmap with colorbar
        cax = ax.imshow(Z, 
            aspect='equal', 
            interpolation='nearest',
            cmap=cm.viridis,
            origin='lower',
            extent=[0, 1, 0, 1])

        # Annotations
        ax.set_title('Greedy state-values')
        ax.set_xlabel('Position')
        ax.set_ylabel('Velocity')
        
        # Set tick labels (subsampling the x-y values, rounding to two decimals)
        nticks  = 8
        ax.set_xticks(np.linspace(0, 1, nticks))
        ax.set_yticks(np.linspace(0, 1, nticks))
        ax.set_xticklabels(np.round(subsample(xx, nticks), 2))
        ax.set_yticklabels(np.round(subsample(yy, nticks), 2))
        cbar = fig.colorbar(cax)
        fig.savefig('test.png')
        plt.show()
