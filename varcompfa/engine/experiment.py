
import numpy as np

import varcompfa as vcf


class PolicyEvaluation:
    """Policy evaluation experiment class."""
    def __init__(self, env, policy, agents=list()):
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
        self.env = env
        self.policy = policy
        self.agents = agents

    def run(self, num_episodes, max_steps, callbacks=list()):
        """Run an experiment.

        Parameters
        ----------
        num_episodes: int
            The number of episodes to run the experiment for.
        max_steps: int
            The maximum number of steps allowed in an episode.
        callbacks: list
            A list of callbacks, objects that may perform actions at certain
            phases of the experiment's execution. (See `varcompfa.callbacks`)
        """
        # Start of experiment callbacks
        for cbk in callbacks:
                cbk.on_experiment_begin()

        # Run for `num_episodes`
        for episode_ix in range(num_episodes):
            # Start of episode callbacks
            for cbk in callbacks:
                cbk.on_episode_begin(episode_ix)

            # Reset the environment
            obs = self.env.reset()

            # Run for at most `max_steps` iterations
            for step_ix in range(max_steps):
                # Perform callbacks for beginning of step
                for cbk in callbacks:
                    cbk.on_step_begin(step_ix)

                action = self.policy.act(obs)
                obs_p, reward, done, info = self.env.step(action)

                # Get the basic context from the current time step
                ctx = {
                    'obs': obs,
                    'obs_p': obs_p,
                    'a': action,
                    'r': reward,
                    'done': done,
                }

                # Perform learning for each of the agents
                for agent in self.agents:
                    agent.update(ctx)

                # Prepare for next iteration
                obs = obs_p

                # Perform callbacks for end of step
                for cbk in callbacks:
                    cbk.on_step_end(step_ix)

                # If terminal state reached, perform a final update?
                if done:
                    break

            else:
                # Failed to terminate in fewer than `max_steps`.
                pass
            # End of episode, either due to terminating or running out of steps
            # Perform end of episode callbacks
            for cbk in callbacks:
                cbk.on_episode_end(episode_ix)

        # Perform end of experiment callbacks
        for cbk in callbacks:
            cbk.on_experiment_end()
