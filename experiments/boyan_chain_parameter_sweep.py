
# coding: utf-8

# In[1]:

# Logging
import logging
logger = logging.getLogger(__name__)
basic_formatter = logging.Formatter('[%(asctime)s] %(message)s')
handler = logging.StreamHandler()
handler.setFormatter(basic_formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


# In[2]:

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')
get_ipython().magic('matplotlib inline')

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

from IPython.display import display


# In[3]:

# Set print options for reporting
np.set_printoptions(suppress=True, precision=6)


# In[4]:

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

def parameter_search(base, vary):
    names, values = zip(*vary.items())
    for case in itertools.product(*values):
        yield {**base, **dict(zip(names, case))}
        
def summarize(arr, axis=0):
    ret = {
        'mean': np.mean(arr, axis=axis),
        'var' : np.var(arr, axis=axis),
        'min' : np.min(arr, axis=axis),
        'max' : np.max(arr, axis=axis)
    }
    return ret


# In[5]:

class BoyanFeatures(vcf.features.Feature):
    __mapping = {
         0 : np.array([0.0 , 0.0 , 0.0 , 0.0 ]),
         1 : np.array([0.0 , 0.0 , 0.0 , 1.0 ]),
         2 : np.array([0.0 , 0.0 , 0.25, 0.75]),
         3 : np.array([0.0 , 0.0 , 0.5 , 0.5 ]),
         4 : np.array([0.0 , 0.0 , 0.75, 0.25]),
         5 : np.array([0.0 , 0.0 , 1.0 , 0.0 ]),
         6 : np.array([0.0 , 0.25, 0.75, 0.0 ]),
         7 : np.array([0.0 , 0.5 , 0.5 , 0.0 ]),
         8 : np.array([0.0 , 0.75, 0.25, 0.0 ]),
         9 : np.array([0.0 , 1.0 , 0.0 , 0.0 ]),
        10 : np.array([0.25, 0.75, 0.0 , 0.0 ]),
        11 : np.array([0.5 , 0.5 , 0.0 , 0.0 ]),
        12 : np.array([0.75, 0.25, 0.0 , 0.0 ]),
        13 : np.array([1.0 , 0.0 , 0.0 , 0.0 ]),      
    }
    def __init__(self, *args, **kwargs):
        self.mapping = {k: v.copy() for k, v in self.__mapping.items()}

    def __call__(self, obs):
        return self.mapping[obs]
    
    def __len__(self):
        return 4
    
    def as_matrix(self):
        num_states = len(self.mapping)
        ret = np.empty((num_states, len(self)))
        for i in range(num_states):
            ret[i] = self.mapping[i]
        return ret


# In[6]:

env = gym.make('BoyanChainMDP-v0')
phi = BoyanFeatures()

num_states = env.observation_space.n
states = np.arange(num_states)
fmat   = phi.as_matrix()

control = vcf.policies.ConstantPolicy(0)


# In[ ]:

pkgdir = os.path.dirname(os.path.dirname(vcf.__file__))
NUM_RUNS = 30
NUM_EPISODES = 40000
OUTDIR = os.path.join(pkgdir, 'results', 'boyan_sweep')
NAME_FMT = "lambda={lmbda:1.2f}_kappa={kappa:1.2f}_lambda_bar={lmbda_bar:1.2f}"


# Parameters to search over
base = {
    'alpha': 0.001,
    'alpha': 0.1,
    'gamma': 1.0,
}

vary = {
    'lmbda': [0.0, 0.9, 1.0],
    'lmbda_bar': [0, 0.9, 1.0],
    'kappa': [0, 0.9, 1.0]
}



for params in parameter_search(base, vary):
    _kappa = params['kappa']
    _gamma = params['gamma']
    _lmbar = params['lmbda_bar']
    
    # Basename for output files
    basename = NAME_FMT.format(**params)
    
    value_params = {
        'alpha': params['alpha'],
        'gm': params['gamma'],
        'gm_p': vcf.Constant(params['gamma'], 0),
        'lm': params['lmbda'],
        'lm_p': params['lmbda'],
    }
    
    direct_params = {
        'alpha': params['alpha'],
        'gm'   : (params['gamma']*params['kappa'])**2,
        'gm_p' : vcf.Constant((params['gamma']*params['kappa'])**2, 0),
        'lm'   : params['lmbda_bar'],
        'lm_p' : params['lmbda_bar'],
    }

    second_params = {
        'alpha': params['alpha'],
        'gm'   : (params['gamma']*params['kappa'])**2,
        'gm_p' : vcf.Constant((params['gamma']*params['kappa'])**2, 0),
        'lm'   : params['lmbda_bar'],
        'lm_p' : params['lmbda_bar'],
    }
    
    # Direct variance algorithm
    def direct_reward(ctx):
        return value_agent.get_td_error(ctx)**2
    
    # Second moment algorithm
    def second_moment_reward(ctx):
        # Get information from the value agent
        v_gm = _gamma
        v_gm_p = value_params['gm_p'](ctx)    
        v_nxt = value_agent.get_value(ctx['obs_p'])

        # For estimating different lambda-return than the value function 
        v_lm = _kappa
        v_lm_p = _kappa

        # Compute next "reward"
        g_bar = ctx['r'] + v_gm_p * (1-v_lm_p) * v_nxt
        r_bar = g_bar**2 + 2*v_gm_p * v_lm_p * g_bar * v_nxt
        return r_bar
    
    
    frames = []
    for run in range(NUM_RUNS):
        value_agent = vcf.Agent(vcf.algos.TD(len(phi)), phi, value_params)
        direct_agent = vcf.Agent(vcf.algos.TD(len(phi)), phi, direct_params,
                                reward_func=direct_reward)
        second_agent = vcf.Agent(vcf.algos.TD(len(phi)), phi, second_params,
                                reward_func=second_moment_reward)
        
        
        # Set experiment
        learners = [direct_agent, second_agent, value_agent]
        experiment = vcf.LiveExperiment(env, control, learners=learners)
        
        # Set up callbacks to record runs
        exclusions = ['x', 'xp']
        value_hist = vcf.callbacks.AgentHistory(value_agent, exclude=exclusions, 
                                                compute={'v_p' : lambda x: value_agent.get_value(x['obs_p']),
                                                         'w_value': lambda x: value_agent.algo.weights,
                                                         'w_direct': lambda x: direct_agent.algo.weights,
                                                         'w_second': lambda x: second_agent.algo.weights,})

        # Set the callbacks and run the experiment
        callbacks = [value_hist]
        experiment.run(NUM_EPISODES, callbacks=callbacks)
        
        # Convert the run's history to a DataFrame
        df = load_df(value_hist.contexts)
        
        # Process dataframe
        df['delta'] = df['update_result'].apply(lambda x: x['delta'])
        df['delta_sq'] = df['delta']**2
        
        # Compute relevant quantities
        df['G'] = vcf.analysis.calculate_return(df['r'], df['gm_p'])
        df['td_values']  = df['w_value'].apply(lambda x: np.dot(fmat, x))
        df['direct_variances'] = df['w_direct'].apply(lambda x: np.dot(fmat, x))
        df['second_values'] = df['w_second'].apply(lambda x: np.dot(fmat, x))
        df['second_variances'] = df['second_values'] - df['td_values']**2
        
        # Compute lambda and kappa returns (using next value, not asymptotic value function)
        # TODO: Here `kappa` and `lambda` are treated as constant, not from dataframe
        df['G_lm'] = vcf.analysis.calculate_lambda_return(df.r, df.gm_p, params['lmbda'], df.v_p)
        df['G_kp'] = vcf.analysis.calculate_lambda_return(df.r, df.gm_p, params['kappa'], df.v_p)
        
        # Remove some columns to reduce memory use
        df.drop(['a', 't', 'total_steps', 'update_result', 'done', 'alpha'], axis=1, inplace=True)
        
        # Preserve the run dataframe
        frames.append(df)
        
    # Collect dataframes from all runs
    run_df = pd.concat(frames, keys=range(NUM_RUNS), names=['run', 't'])
        
    # Perform analysis for Craig
    res = {
        'td': defaultdict(list),
        'second_moment': defaultdict(list),
        'direct': defaultdict(list),
    }
    
    # Group by episode
    grouped = run_df.groupby('episode')

    # Summarize over the runs
    for i, episode in grouped:
        for key, val in summarize(np.vstack(episode['td_values'])).items():
            res['td'][key].append(val)
            
        for key, val in summarize(np.vstack(episode['direct_variances'])).items():
            res['direct'][key].append(val)
            
        for key, val in summarize(np.vstack(episode['second_variances'])).items():
            res['second_moment'][key].append(val)
     
    # ############################################################
    # TODO: Make this neater, use a DataFrame
    # Print the averaged mean, direct and second moment results
    # ############################################################
    val_mean = np.mean(res['td']['mean'], axis=0)
    dir_mean = np.mean(res['direct']['mean'], axis=0)
    sec_mean = np.mean(res['second_moment']['mean'], axis=0)

    print("Average Values:")
    print(val_mean)
    print(dir_mean)
    print(sec_mean)
    
    print("Final Values:")
    print(np.array([value_agent.get_value(s) for s in states]))
    print(np.array([direct_agent.get_value(s) for s in states]))
    print(np.array([second_agent.get_value(s) - value_agent.get_value(s)**2 for s in states]))
    
    
    # Convert to Craig's format
    for name, dct in res.items():
        for key, val in dct.items():
            dct[key] = np.array(val).T.tolist()
    
    
    ##############################################################
    # Plotting
    ##############################################################
    fig, axes = plt.subplots(1,3)
    
    for x in np.array(res['td']['mean']):
        axes[0].plot(x)
        
    for x in np.array(res['direct']['mean']):
        axes[1].plot(x)
        
    for x in np.array(res['second_moment']['mean']):
        axes[2].plot(x)
    
    # Formatting
    fig.set_size_inches(12, 6)

    # Titles
    axes[0].set_title("Values")
    axes[1].set_title("Direct")
    axes[2].set_title("Second Moment")
    
    # Label axes
    axes[0].set_xlabel('Episode')
    axes[1].set_xlabel('Episode')
    axes[2].set_xlabel('Episode')
    
    # Common axes
    axes[1].set_ylim(-50, 250)
    axes[2].set_ylim(-50, 250)
    
    # Save the figure
    os.makedirs(os.path.join(OUTDIR, 'plots'), exist_ok=True)
    plot_path = os.path.join(OUTDIR, 'plots', basename + '.png')
    fig.savefig(plot_path)
    
    # Zooming in on variance
    #############################################################
    fig, axes = plt.subplots(1,3)
    
    for x in np.array(res['td']['mean']):
        axes[0].plot(x)
        
    for x in np.array(res['direct']['mean']):
        axes[1].plot(x)
        
    for x in np.array(res['second_moment']['mean']):
        axes[2].plot(x)
    
    # Formatting
    fig.set_size_inches(12, 6)

    # Titles
    axes[0].set_title("Values")
    axes[1].set_title("Direct")
    axes[2].set_title("Second Moment")
    
    # Label axes
    axes[0].set_xlabel('Episode')
    axes[1].set_xlabel('Episode')
    axes[2].set_xlabel('Episode')

    # Zoom in
    axes[1].set_ylim(0, 10)
    axes[2].set_ylim(0, 10)
    
    # Save the figure
    os.makedirs(os.path.join(OUTDIR, 'plots'), exist_ok=True)
    plot_path = os.path.join(OUTDIR, 'plots', basename + '-zoom' + '.png')
    fig.savefig(plot_path)
    
    # Avoid displaying graphs
    plt.close()
    
    
    # Record some metadata
    dct['metadata'] = {
        'num_runs': NUM_RUNS,
        'num_episodes': NUM_EPISODES,
        'environment': env.spec.id,
        'params': params
    }
    
    # Save result
    os.makedirs(OUTDIR, exist_ok=True)
    result_path = os.path.join(OUTDIR, 'results', basename + '.json')
    print("Saving results to:", result_path)
    os.makedirs(os.path.join(OUTDIR, 'results'), exist_ok=True)
    json.dump(res, open(result_path, 'w'))


# In[ ]:



