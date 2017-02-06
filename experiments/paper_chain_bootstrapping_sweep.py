"""
An experiment that performs a parameter sweep over λ, κ, and \\bar{λ}.

Not exactly the same data format as the rest of the experiments because it's for
a paper, and collaborators deserve to get the results in the format they prefer.
"""
# Logging
import logging
logger = logging.getLogger(__name__)
basic_formatter = logging.Formatter('[%(asctime)s] %(message)s')
handler = logging.StreamHandler()
handler.setFormatter(basic_formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

import itertools
import json
import os
import gym
import matplotlib as mpl
import matplotlib.pyplot as plt
import json_tricks as jt
import networkx as nx
import numpy as np
import pandas as pd
import varcompfa as vcf
from collections import defaultdict

# Set print options for reporting
np.set_printoptions(suppress=True, precision=6)

# Helper functions
def load_df(contexts):
    """Load contexts into a DataFrame with some preprocessing"""
    ret = pd.DataFrame(contexts)
    from numbers import Number

    def make_hashable(elem):
        if isinstance(elem, Number):
            return elem
        elif isinstance(elem, np.ndarray) and elem.ndim == 0:
            return elem.item()
        else:
            return tuple(elem)

    # Make it possible to hash (and therefore group) certain columns
    ret['obs'] = ret['obs'].apply(make_hashable)
    ret['obs_p'] = ret['obs_p'].apply(make_hashable)

    return ret

def summarize(arr, axis=0):
    """Compute some statistics over an array.

    Here, we assume the array is a 2D array, with shape
        `(num_steps, num_states)`,

    If the input array contains value functions, then each item (i,j)
    represents the value of state `j` at time `i` in an episode.
    """
    ret = {
        'mean': np.mean(arr, axis=axis),
        'var' : np.var(arr, axis=axis),
        'min' : np.min(arr, axis=axis),
        'max' : np.max(arr, axis=axis)
    }
    return ret

def parameter_search(base, vary=dict()):
    """A quick parameter search generator function.

    Assumes that `base` consists of `(key, value)` pairs which remain fixed,
    while `vary` consists of `(key, Seq[value])` pairs.
    Returns an iterable of dicts with each item having the base values fixed
    and an element of the cartesian product of the values in `vary`.
    """
    names, values = zip(*vary.items())
    for case in itertools.product(*values):
        yield {**base, **dict(zip(names, case))}


# The actual experiment
if __name__ == "__main__":
    # Number of runs and episodes per-run
    num_runs = 30
    num_episodes = 30000
    ENV_SEED = 237

    # Set output directory and name format
    outdir = "fig-7-8-9-results"
    name_fmt = "lambda={lmbda:1.2f}_kappa={kappa:1.2f}_lambda_bar={lmbda_bar:1.2f}"
    os.makedirs(outdir, exist_ok=True)

    # Setup the environment with a specified seed
    env = gym.make('PaperChainMDP-v0')
    env.seed(ENV_SEED)
    phi = vcf.features.BinaryVector(env.observation_space.n)

    # Compute some relevant things
    num_states = len(phi)
    states = np.arange(num_states)
    fmat = np.array([phi(s) for s in states])

    # Behavior policy
    control = vcf.policies.ConstantPolicy(0)

    # Parameters to search over
    base = {
        'alpha': 0.001,
        'gamma': 1.0,
    }

    vary = {
        'lmbda': [0.0, 0.9, 1.0],
        'lmbda_bar': [0, 0.9, 1.0],
        'kappa': [0, 0.9, 1.0]
    }

    for params in parameter_search(base, vary):
        gamma = params['gamma']
        lmbda = params['lmbda']
        kappa = vcf.Constant(params['kappa'])
        kappa_p = vcf.Constant(params['kappa'], 0)
        lmbda_bar = params['lmbda_bar']

        # Specify the parameters for the agents
        value_params = {
            'alpha': params['alpha'],
            'gm' : vcf.Constant(gamma),
            'gm_p': vcf.Constant(gamma, 0),
            'lm': lmbda,
            'lm_p': lmbda,
        }

        direct_params = {
            'alpha': params['alpha'],
            'gm'   : lambda x: (value_params['gm'](x)*kappa(x))**2,
            'gm_p' : lambda x: (value_params['gm_p'](x)*kappa_p(x))**2,
            'lm'   : lmbda_bar,
            'lm_p' : lmbda_bar,
        }

        second_params = {
            'alpha': params['alpha'],
            'gm'   : lambda x: (value_params['gm'](x)*kappa(x))**2,
            'gm_p' : lambda x: (value_params['gm_p'](x)*kappa_p(x))**2,
            'lm'   : lmbda_bar,
            'lm_p' : lmbda_bar,
        }

        # Direct variance algorithm
        def direct_reward(ctx):
            return value_agent.get_td_error(ctx)**2

        # Second moment algorithm
        def second_moment_reward(ctx):
            # Get information from the value agent
            v_gm = value_params['gm'](ctx)
            v_gm_p = value_params['gm_p'](ctx)
            # For estimating different lambda-return than the value function
            v_lm = kappa(ctx)
            v_lm_p = kappa_p(ctx)
            v_nxt = value_agent.get_value(ctx['obs_p'])

            # Compute next "reward"
            g_bar = ctx['r'] + v_gm_p*(1-v_lm_p)*v_nxt
            r_bar = g_bar**2 + 2*v_gm_p*v_lm_p*g_bar*v_nxt
            return r_bar

        # Set up storage for run data
        frames = []

        # Perform multiple runs of the given number of episodes
        for run in range(num_runs):
            # Set up (or reset) the agents
            value_agent  = vcf.Agent(vcf.algos.TD(num_states), phi, value_params)
            direct_agent = vcf.Agent(vcf.algos.TD(num_states), phi, direct_params,
                                     reward_func=direct_reward)
            second_agent = vcf.Agent(vcf.algos.TD(num_states), phi, second_params,
                                     reward_func=second_moment_reward)


            # Set up the experiment
            learners = [direct_agent, second_agent, value_agent]
            experiment = vcf.LiveExperiment(env, control, learners=learners)

            # Set up callbacks to record runs
            exclusions = ['x', 'xp']
            value_hist = vcf.callbacks.AgentHistory(value_agent, exclude=exclusions,
                                                    compute={'v': lambda x: value_agent.get_value(x['obs']),
                                                             'v_p' : lambda x: value_agent.get_value(x['obs_p']),
                                                             'weights': lambda x: value_agent.algo.weights})
            direct_hist = vcf.callbacks.AgentHistory(direct_agent, exclude=exclusions,
                                                    compute={'weights': lambda x: direct_agent.algo.weights})
            second_hist = vcf.callbacks.AgentHistory(second_agent, exclude=exclusions,
                                                    compute={'weights': lambda x: second_agent.algo.weights})
            callbacks = [value_hist, direct_hist, second_hist]

            # Run the experiment
            experiment.run(num_episodes, callbacks=callbacks)

            # Load the data for analysis
            df = load_df(value_hist.contexts)


            # Compute relevant quantities
            df['G'] = vcf.analysis.calculate_return(df['r'], df['gm_p'])
            df['td_values']  = df['weights'].apply(lambda x: np.dot(x, fmat))

            second_weights = [x['weights'] for x in second_hist.contexts]
            df['second_values'] = [np.dot(x, fmat) for x in second_weights]
            df['second_variances'] = df['second_values'] - df['td_values']**2

            direct_weights = [x['weights'] for x in direct_hist.contexts]
            df['direct_variances'] = [np.dot(x, fmat) for x in direct_weights]

            # Compute lambda and kappa returns (using next value, not asymptotic value function)
            # TODO: Here `kappa` and `lambda` are treated as constant, not from dataframe
            df['G_lm'] = vcf.analysis.calculate_lambda_return(df.r, df.gm_p, lmbda, df.v_p)
            df['G_kp'] = vcf.analysis.calculate_lambda_return(df.r, df.gm_p, kappa.value, df.v_p)


            # TESTING
            # sdf = load_df(second_hist.contexts)
            # df['r_bar'] = sdf['r']
            # df['lm_bar_p'] = sdf['lm_p']

            # Preserve the run in the panel
            frames.append(df)

        # Collect dataframes from all runs
        run_df = pd.concat(frames, keys=range(num_runs), names=['run', 't'])


        # Perform analysis for Craig
        res = {
            'td': defaultdict(list),
            'second_moment': defaultdict(list),
            'direct': defaultdict(list),
        }

        ge = run_df.groupby('episode')
        for i, episode in ge:

            # Get summary across all runs for value function estimate
            vv = np.vstack(episode['td_values'])
            vr = summarize(vv)
            for key, val in vr.items():
                res['td'][key].append(val)

            # Summarize second moment method
            vs = np.vstack(episode['second_variances'])
            for key, val in summarize(vs).items():
                res['second_moment'][key].append(val)

            # Summarize direct method
            vd = np.vstack(episode['direct_variances'])
            for key, val in summarize(vd).items():
                res['direct'][key].append(val)


        # Print the averaged mean, direct and second moment results
        val_mean = np.mean(res['td']['mean'], axis=0)
        dir_mean = np.mean(res['direct']['mean'], axis=0)
        sec_mean = np.mean(res['second_moment']['mean'], axis=0)

        # Reporting
        print("Average Values:")
        print(val_mean)
        print(dir_mean)
        print(sec_mean)

        print("Final Values:")
        print(np.array([value_agent.get_value(s) for s in states]))
        print(np.array([direct_agent.get_value(s) for s in states]))
        print(np.array([second_agent.get_value(s) - value_agent.get_value(s)**2 for s in states]))


        # Plotting
        # Check that graphs look OK
        fig, axes = plt.subplots(1,3)
        fig.set_size_inches(12, 6)

        for x in np.array(res['td']['mean']).T:
            axes[0].plot(x)

        for x in np.array(res['direct']['mean']).T:
            axes[1].plot(x)

        for x in np.array(res['second_moment']['mean']).T:
            axes[2].plot(x)

        # Formatting of graph
        axes[1].set_ylim(0, 8)
        axes[2].set_ylim(0, 8)

        # Convert results to Craig's format
        for name, dct in res.items():
            for key, val in dct.items():
                dct[key] = np.vstack(val).T.tolist()

        # Metadata for the experiment.
        dct['metadata'] = {
            'num_runs': num_runs,
            'num_episodes': num_episodes,
            'environment': env.spec.id,
            'params': params
        }


        filename = name_fmt.format(**params)
        print("Saving results to:", filename)

        result_path = os.path.join(outdir, filename + '-result.json')
        json.dump(res, open(result_path, 'w'))

        fig_path = os.path.join(outdir, filename+'.png')
        # fig.savefig(fig_path)
        plt.close()
