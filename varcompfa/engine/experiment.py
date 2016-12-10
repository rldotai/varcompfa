"""
Code for running experiments in a reproducible way.
"""
import datetime
import logging
import os
import time
import numpy as np

import varcompfa as vcf

logger = logging.getLogger(__name__)


def _time_string(fmt=None):
    """Get a string representation of the current time, with the option to
    specify a different format."""
    if fmt is None:
        fmt = "%Y-%m-%d-%H-%M-%S-%f"
    return datetime.datetime.now().strftime(fmt)

def _make_experiment_dir(basedir=None, target=None):
    """Make a directory to store data generated by experiment."""
    if basedir is None:
        basedir = os.curdir
    if target is None:
        target = _time_string()
    path = os.path.join(os.path.abspath(basedir), target)
    # os.makedirs(path, exist_ok=True)
    return path


class PolicyEvaluation:
    """Policy evaluation experiment class."""
    def __init__(self, environment, policy, agents=list()):
        """Create an experiment

        Parameters
        ----------
        env : gym.Env
            The environment on which to run the experiment
        policy : object
            An object that has an `act` function, which produces a valid action
            for the environment given an observation.
        agents: list
            A list of agents to update at each timestep
        """
        self.env = environment
        self.policy = policy
        self.agents = agents

    def run(self, num_episodes, max_steps, callbacks=list()):
        """Run an experiment.

        Recording the results of the experiment can be done via `Callback`
        classes, which get called at certain times throughout the run.
        These classes pass `dict` objects containing information to the
        callbacks.

        Parameters
        ----------
        num_episodes: int
            The number of episodes to run the experiment for.
        max_steps: int
            The maximum number of steps allowed in an episode.
        callbacks: list
            A list of callbacks, objects that may perform actions at certain
            phases of the experiment's execution. (See `varcompfa.callbacks`)


        Callback Details
        ----------------
        A single callback object can have different methods which get called at
        different phases of the run's execution.


        - `on_experiment_begin()`
            + Called once per-run, at the start of the experiment.
        - `on_experiment_end()`
            + Called once per-run, at the end of the experiment.
        - `on_episode_begin()`
            + Called once-per episode, prior to the start of the episode.
        - `on_experiment_end()`
            + Called once-per episode, at the end of the episode.
        - `on_step_begin()`
            + Called before executing every step of every episode
        - `on_step_end()`
            + Called at the end of every step of every episode
        """
        # Information that should be generally available
        run_params = {
            'environment': self.env,
            'policy': self.policy,
            'agents': self.agents,
        }

        # Start of experiment callbacks
        run_begin_info = {
            **run_params,
            'num_episodes': num_episodes,
            'max_steps': max_steps,
            'version': vcf.utils.current_version(),
            'git_hash': vcf.utils.current_git_hash(),
            'start_time': time.time(),
        }
        for cbk in callbacks:
                cbk.on_experiment_begin(run_begin_info)

        # Track total number of steps
        total_steps = 0
        # Run for `num_episodes`
        for episode_ix in range(num_episodes):
            # Start of episode callbacks
            episode_begin_info = {'total_steps' : total_steps}
            for cbk in callbacks:
                cbk.on_episode_begin(episode_ix, episode_begin_info)

            # Reset the environment, get initial observation
            obs = self.env.reset()
            # Run for at most `max_steps` iterations
            for step_ix in range(max_steps):
                # Perform callbacks for beginning of step
                step_begin_info = {}
                for cbk in callbacks:
                    cbk.on_step_begin(step_ix, step_begin_info)

                action = self.policy.act(obs)
                obs_p, reward, done, info = self.env.step(action)

                # Get the basic context from the current time step
                ctx = {
                    'total_steps' : total_steps,
                    'obs': obs,
                    'obs_p': obs_p,
                    'a': action,
                    'r': reward,
                    'done': done,
                }

                # Perform learning for each of the agents
                update_results = []
                for agent in self.agents:
                    update_results.append(agent.update(ctx))

                # Prepare for next iteration
                obs = obs_p
                total_steps += 1

                # Perform callbacks for end of step
                step_end_info = {
                    'context': ctx,
                    'update_results': update_results,
                }
                for cbk in callbacks:
                    cbk.on_step_end(step_ix, step_end_info)

                # If terminal state reached, exit episode loop
                if done:
                    break

            else:
                # Failed to terminate in fewer than `max_steps`.
                pass
            # End of episode, either due to terminating or running out of steps
            # Perform end of episode callbacks
            episode_end_info = {
                'total_steps' : total_steps,
            }
            for cbk in callbacks:
                cbk.on_episode_end(episode_ix, episode_end_info)

        # Perform end of experiment callbacks
        experiment_end_info = {
            'total_steps' : total_steps,
        }
        for cbk in callbacks:
            cbk.on_experiment_end(experiment_end_info)
