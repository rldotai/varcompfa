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

# The package itself
import varcompfa as vcf 
from varcompfa.misc.analysis import *


if __name__ == "__main__":
    # Specify experiment
    num_episodes = 10
    max_steps = 3000

    # Set up the experiment
    env = gym.make('SimpleMDP-v0')

    # Set up representation
    phi = vcf.BinaryVector(env.observation_space.n)

    # Set up agent
    nf = len(phi)
    na = env.action_space.n
    agent = vcf.DiscreteQ(nf, na, epsilon=0.05)

    # Zero value initialization
    agent.w *= 0
    # Fixed parameters
    alpha = 0.1
    gamma = 0.999
    lmbda = 0.8

    # Set up tracking
    episodes = []
    stepcount = []
    for i in range(num_episodes):
        # reset environment and get initial feature
        obs = env.reset()
        x = phi(obs)
        # current episode tracking
        history = []
        for j in range(max_steps):
            action = agent.act(x)
            obs_p, reward, done, info = env.step(action)
            xp = phi(obs_p)

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

            # set up next iteration
            x = xp

            # exit if done, otherwise set up for next iteration
            if done:
                logger.info("Reached terminal state in %d steps"%(j+1))
                break
        else:
            logger.info("Failed to reach the end before the time limit")

        # log end of episode information
        episodes.append(history)
        stepcount.append(j)