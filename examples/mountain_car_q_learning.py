import numpy as np 
import gym

# Analysis
from numpy.linalg import pinv
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd 
import scipy as sp 
import scipy.linalg, scipy.sparse, scipy.stats
import itertools
import toolz
from toolz import concat, pluck


# Logging
import logging
# get a logger, set the logging level
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# The package itself
import varcompfa as vcf 
from varcompfa.misc.analysis import *


if __name__ == "__main__":
    # Specify experiment
    num_episodes = 1000
    max_steps = 3000

    # Set up the experiment
    env = gym.make('MountainCar-v0')

    # Set up representation
    bias_unit   = vcf.BiasUnit()
    # Tile coding for discretization
    tiling_1    = vcf.UniformTiling(env.observation_space, 3)
    tiling_2    = vcf.UniformTiling(env.observation_space, 5)
    tiling_3    = vcf.UniformTiling(env.observation_space, 11)
    tiling_4    = vcf.UniformTiling(env.observation_space, 19)
    # Convert tile indices to binary vector
    bvec_1      = vcf.BinaryVector(tiling_1.high, tiling_1) 
    bvec_2      = vcf.BinaryVector(tiling_2.high, tiling_2)
    bvec_3      = vcf.BinaryVector(tiling_3.high, tiling_3)
    bvec_4      = vcf.BinaryVector(tiling_4.high, tiling_4)
    # Concatenate binary vectors
    phi         = vcf.Union(bias_unit, bvec_1, bvec_2, bvec_3, bvec_4)



    # Set up agents
    nf = len(phi)
    na = env.action_space.n
    # Control agent, value function learning, delta agent, delta-squared agent
    control_agent   = vcf.DiscreteQ(nf, na, epsilon=0.05) 
    value_agent     = vcf.TD(nf)
    delta_agent     = vcf.TD(nf)
    square_agent    = vcf.TD(nf)

    # Zero value initialization
    control_agent.w *= 0
    control_agent.w += np.random.normal(0, 1, control_agent.w.shape)

    # Fixed parameters
    alpha_0 = 0.05
    gamma = 0.999
    lmbda = 0.0

    # Set up tracking
    episodes = []
    stepcount = []
    for i in range(num_episodes):
        # decrease learning rate
        # CRAIG: why are you affecting alpha this way?
        alpha = alpha_0/(1 + alpha_0**(1/2))
        # reset environment and get initial feature
        obs = env.reset()
        x = phi(obs)
        # current episode tracking
        history = []
        for j in range(max_steps):
            action = control_agent.act(x)
            obs_p, reward, done, info = env.step(action)
            xp = phi(obs_p)

            # update learning algorithms
            # CRAIG: gm and gm_p makes me nervous. Maybe it doesn't matter, but I feel like the learning
            # agent should track what it's last gamma was all by itself. It shouldn't be a parameter that you
            # provide.
            gm    = gamma
            gm_p  = gamma if not done else 0
            lm    = lmbda
            control_delta   = control_agent.learn(x, action, reward, xp, alpha, gm, lm)
            value_delta     = value_agent.learn(x, reward, xp, alpha, gm, gm_p, lm)
            # CRAIG: why are you tracking delta as a cumulant?
            delta_delta     = delta_agent.learn(x, value_delta, xp, alpha, gm, gm_p, lm)
            square_delta    = square_agent.learn(x, value_delta**2, xp, alpha, gm, gm_p, lm)

            # log information about the timestep
            history.append(dict(
                obs=obs.copy(),
                action=action, 
                reward=reward,
                done=done,
                # CRAIG: ...this could be big
                x=x.copy(),
                control_delta=control_delta,
                value_delta=value_delta,
                delta_delta=delta_delta,
                square_delta=square_delta,
                alpha=alpha,
                gm=gm,
                gm_p=gm_p,
                lm=lmbda,
            ))

            # set up next iteration
            x   = xp
            obs = obs_p

            # exit if done, otherwise set up for next iteration
            if done:
                # CRAIG: this feels weird to me to perform this one last step. I assume this is a bit of a quirk
                # from the way OpenGym implemented things? So x is the penultimate terminal state, not the terminal one?
                # perform final update
                xp = np.zeros_like(x)
                reward = 0
                gm = 0
                gm_p  = 0
                lm = 0
                control_delta   = control_agent.learn(x, action, reward, xp, alpha, gm, lm)
                value_delta     = value_agent.learn(x, reward, xp, alpha, gm, gm_p, lm)
                delta_delta     = delta_agent.learn(x, value_delta, xp, alpha, gm, gm_p, lm)
                square_delta    = square_agent.learn(x, value_delta**2, xp, alpha, gm, gm_p, lm)
                 
                # log information about the timestep
                history.append(dict(
                    obs=obs_p.copy(),
                    action=action, 
                    reward=reward,
                    done=done,
                    x=x.copy(),
                    control_delta=control_delta,
                    value_delta=value_delta,
                    delta_delta=delta_delta,
                    square_delta=square_delta,
                    alpha=alpha,
                    gm=gm,
                    gm_p=gm_p,
                    lm=lm,
                ))

                # Decrease exploration
                control_agent.epsilon *= 0.99

                logger.info("Reached terminal state in %d steps"%(j+1))
                break
        else:
            logger.info("Failed to reach the end before the time limit")

        # log end of episode information
        episodes.append(history)
        stepcount.append(j)

    # Truncate to last few episodes
    episodes = episodes[-100:]

    # # Calculate return and least-squares feature weights
    # xa              = apluck('x', concat(episodes))
    # ga              = np.hstack([episode_return(ep) for ep in episodes])
    # w_ls, *_        = np.linalg.lstsq(xa, ga)
    # # Calculate variance via residuals, and the per-feature variance
    # ghat_ls         = np.dot(xa, w_ls)
    # var_ls          = (ga - ghat_ls)**2
    # var_w_ls, *_    = np.linalg.lstsq(xa, var_ls)


    # # 2D-Heatmap
    # traj = np.array(list(toolz.pluck('obs', episodes[-1])))   
    # xlim, ylim = list(zip(traj.min(axis=0), traj.max(axis=0)))  
    # xdata = np.linspace(*xlim)
    # ydata = np.linspace(*ylim)
    
    # # Compute grid locations
    # shape = (len(xdata), len(ydata))
    # indices = np.ndindex(shape)
    # grid = np.array([(xdata[i], ydata[j]) for i, j in indices])
    # data = np.array([phi(i) for i in grid])
    
    # # Compute value functions for data
    # Z_lsret = np.dot(data, w_ls)
    # Z_value = np.dot(data, value_agent.w)
    # Z_delta = np.dot(data, delta_agent.w)

    # # Reshape the data 
    # # TODO: Is there a better way of generating the data to get to this step?
    # Z_lsret = Z_lsret.reshape(*shape).T
    # Z_value = Z_value.reshape(*shape).T
    # Z_delta = Z_delta.reshape(*shape).T

    # # Set up plotting function 
    # def custom_heatmap(ax, zmat):
    #     cax = ax.imshow(zmat, 
    #         aspect='equal', 
    #         interpolation='nearest',
    #         cmap=cm.viridis,
    #         origin='lower',
    #         extent=[0, 1, 0, 1])

    #     return cax

    # Actually plot the heat map
    # fig, axes = plt.subplots(2,2)
    # custom_heatmap(axes[0,0], Z_lsret)
    # custom_heatmap(axes[0,1], Z_value)
    # custom_heatmap(axes[1,0], Z_delta)

    # plt.show()
    # np.sum((ga - np.dot(xa, value_agent.w))**2) - var_ls.sum()
    # 
    # def grid_data(xdata, ydata, phi_func):
    #   pass
    #   
    #   
    
    # Multiple approaches to least-squares fitting
    # w_ls, res, rank, svals = np.linalg.lstsq(xa, ga)
    # w_lsqr, *_ = sp.sparse.linalg.lsqr(xa, ga)
    # w_normal = np.linalg.pinv(xa.T @ xa) @ xa.T @ ga
    # Z1 = np.dot(data, w_ls)
    # Z2 = np.dot(data, value_agent.w)
    # Z3 = np.dot(data, w_lsqr)
    # Z4 = np.dot(data, w_normal)

    # Z1 = Z1.reshape(*shape).T
    # Z2 = Z2.reshape(*shape).T
    # Z3 = Z3.reshape(*shape).T
    # Z4 = Z4.reshape(*shape).T

    # # Actually plot the heat map
    # fig, axes = plt.subplots(2,2)
    # custom_heatmap(axes[0,0], Z1)
    # custom_heatmap(axes[0,1], Z2)
    # custom_heatmap(axes[1,0], Z3)
    # custom_heatmap(axes[1,1], Z4)

    # plt.show()

    ###########################################################################
    # Plotting errors
    # history = lconcat(episodes[-100:])[-1000:]
    # fig, axes = plt.subplots(4)
    # axes[0].plot(ga)
    # axes[1].plot(apluck('control_delta', history))
    # axes[2].plot(apluck('value_delta', history))
    # axes[3].plot(apluck('delta_delta', history))

    ###########################################################################
    # Plotting returns
    
    # CRAIG: It's strange to me that you're plotting the estimates in time. I would have
    # expected them to be plotted in state-space instead.
    history = lconcat(episodes[-100:])[-1000:]
    xa = apluck('x', history)
    ra = apluck('reward', history)
    gg = apluck('gm', history)
    ga = calculate_return(ra, gg)

    # Discounted delta-return
    dt = apluck('value_delta', history)
    g_dt = calculate_return(dt, gg)

    w_ls = pinv(xa.T @ xa) @ xa.T @ ga
    # Approximations of return 
    g_ls = np.dot(xa, w_ls)
    g_va = np.dot(xa, value_agent.w)
    g_vd = np.dot(xa, delta_agent.w)

    # CRAIG: where's the plot for the squared delta?
    fig, axes = plt.subplots(5)
    # true return
    axes[0].plot(ga)
    # least-squares estimate of return
    axes[1].plot(g_ls)
    # predicted return
    axes[2].plot(g_va)
    # predicted TD-error
    axes[3].plot(g_vd)
    axes[3].plot(dt)
    # differences between returns
    axes[4].plot(ga - g_va)
    axes[4].plot(g_ls - g_va)
    axes[4].plot(g_vd)
    # show the plots 
    plt.show()

    ###########################################################################
    # zooming in on differences between algorithms
    fig, axes = plt.subplots(2)
    axes[0].plot(ga)
    axes[0].plot(g_va)
    axes[1].plot(ga - g_va)
    axes[1].plot(g_vd)

    plt.show()