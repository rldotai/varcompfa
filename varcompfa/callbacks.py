"""
Callbacks to execute at various points in the course of running an experiment.

Based on Keras' callbacks.
"""
import datetime
import logging
import time
import json_tricks
import pandas as pd
logger = logging.getLogger(__name__)


class Callback:
    """Base class used to build new callbacks.

    For an example of when these callbacks are run, refer to the experiment
    running class `varcompfa.PolicyEvaluation`.
    """
    def __init__(self):
        pass

    def on_experiment_begin(self, info=dict()):
        pass

    def on_experiment_end(self, info=dict()):
        pass

    def on_episode_begin(self, episode_ix, info=dict()):
        pass

    def on_episode_end(self, episode_ix, info=dict()):
        pass

    def on_step_begin(self, step_ix, info=dict()):
        pass

    def on_step_end(self, step_ix, info=dict()):
        pass


class LambdaCallback(Callback):
    """A callback class used to quickly make new custom callbacks."""
    def __init__(self,
                 on_experiment_begin=None,
                 on_experiment_end=None,
                 on_episode_begin=None,
                 on_episode_end=None,
                 on_step_begin=None,
                 on_step_end=None):
        if on_experiment_begin is not None:
            self.on_experiment_begin = on_experiment_begin
        if on_experiment_end is not None:
            self.on_experiment_end = on_experiment_end
        if on_episode_begin is not None:
            self.on_episode_begin = on_episode_begin
        if on_episode_end is not None:
            self.on_episode_end = on_episode_end
        if on_step_begin is not None:
            self.on_step_begin = on_step_begin
        if on_step_end is not None:
            self.on_step_end = on_step_end



class _AnnoyinglyVerboseCallback(Callback):
    """An example callback that pretty-prints all information it has access to
    at each point when it gets called.

    It will likely print a lot.
    """
    def on_experiment_begin(self, info=dict()):
        print("Started training")
        # pprint(info)
        print(info.keys())

    def on_experiment_end(self, info=dict()):
        print("End of training")
        # pprint(info)
        print(info.keys())

    def on_episode_begin(self, episode_ix, info=dict()):
        print("Started episode: %d"%episode_ix)
        # pprint(info)
        print(info.keys())

    def on_episode_end(self, episode_ix, info=dict()):
        print("End of episode: %d"%episode_ix)
        # pprint(info)
        print(info.keys())

    def on_step_begin(self, step_ix, info=dict()):
        print("Begin step: %d"%step_ix)
        # pprint(info)
        print(info.keys())

    def on_step_end(self, step_ix, info=dict()):
        print("End step: %d"%step_ix)
        # pprint(info)
        print(info.keys())


class CheckpointFinal(Callback):
    """Save the agent at the end of the experiment."""
    def __init__(self, agent, filepath):
        self.agent = agent
        self.filepath = filepath

    def on_experiment_end(self, info=dict()):
        logger.info("Saving agent to: %s"%self.filepath)
        # Avoid warning on serializing numpy array scalars
        json_tricks.NumpyEncoder.SHOW_SCALAR_WARNING=False
        json_tricks.dump(self.agent, open(self.filepath, 'w'))
        json_tricks.NumpyEncoder.SHOW_SCALAR_WARNING=True


class CheckpointScheduled(Callback):
    """Checkpoint the agent on a schedule."""
    #TODO: Implement or remove
    pass


class AgentHistory(Callback):
    """
    Record a history of an experiment for a particular agent, using the context
    provided by `update` to get the information the agent used at each step.


    Attributes
    ----------
    history: dict
        The information recorded from the experiment (contexts and metadata).
    contexts: list[dict]
        A list of the contexts returned from the learning agent's update.
        Includes the 'usual' context as well as all the computed quantities the
        agent needed to execute its update.
    metadata: dict
        A dictionary containing metadata about the experiment.


    Notes
    -----

    The history has the form:

        - metadata
            - start_time
            - end_time
            - num_episodes
            - max_steps
            - version
            - git_hash
            - total_time
            - environment
            - control policy
        - contexts
            - total_steps
            - episode
            - t (current timestep in episode)
            - obs
            - obs_p
            - a (action)
            - r (reward)
            - done
            - x (features for `obs`)
            - xp (features for `obs_p`)
            - update_result (the value returned by `agent.update()`
            - weights
            - traces
            - additional parameters computed as part of `agent.update()`


    Remarks
    -------

    Serialization makes things difficult, but some things that are worth
    recording (such as environment and control policy) are hard to serialize.
    At this point I intend to use `pickle` to do this, which solves the problem
    of serialization but introduces potentially weird issues that I have yet to
    investigate.
    For example, does unpickling of an object fail catastrophically if the
    implementation of that object's class changes significantly?

    Currently relies on the learning agents ordering being fixed in order to
    select the right context from those returned by `update_contexts`)
    Alternative implementations are possible, as is modifying the experiment
    class, but those options will only be invoked if I find some pressing
    reason to add/remove/reorder the learning agents over the course of a run.
    """
    def __init__(self, agent, exclude=set(), compute=dict()):
        """
        Initialize the callback.

        Parameters
        ----------
        agent: object
            The agent to record a history for.
        exclude : Set[str], optional
            The set of keys to ignore from `update_context`.
            Useful for when some items, such as the feature vector, may be
            large or otherwise not worth keeping track of.
        compute : Dict, optional
            A dictionary mapping keys to functions which accept a context
            and return a value to track.
        """
        self.agent = agent
        self._exclude = set(exclude)
        self._compute = compute
        self._hist = {}
        self._hist['metadata'] = dict()
        self._hist['contexts'] = list()

    def on_experiment_begin(self, info=dict()):
        # Get the agent's index in the list of learners
        self._agent_ix = info['learners'].index(self.agent)
        self._episode = None

        # Record some metadata
        self._hist['metadata'] = {
            'version'       : info['version'],
            'git_hash'      : info['git_hash'],
            'start_time'    : info['start_time'],
            'num_episodes'  : info['num_episodes'],
            'max_steps'     : info['max_steps'],
            'environment'   : info['environment'],
            'policy'        : info['policy'],
            'learners'      : info['learners'],
        }

    def on_experiment_end(self, info=dict()):
        self._hist['metadata']['num_episodes'] = self._episode + 1
        self._hist['metadata']['end_time'] = info['end_time']
        self._hist['metadata']['total_time'] = \
            (info['end_time'] - self._hist['metadata']['start_time']).total_seconds()

    def on_episode_begin(self, episode_ix, info=dict()):
        self._t = 0

        if self._episode is None:
            self._episode = 0
        else:
            self._episode += 1

    def on_step_end(self, step_ix, info=dict()):
        agent_ctx = info['update_contexts'][self._agent_ix]

        # Preserve the current step's context, ignoring excluded keys
        ctx = {k: v for k, v in agent_ctx.items() if k not in self._exclude}

        # Record weights and traces
        ctx['weights'] = self.agent.algo.weights
        ctx['traces'] = self.agent.algo.traces

        # Compute any additional values that should be tracked
        for k, func in self._compute.items():
            ctx[k] = func(agent_ctx)

        # Combine and append
        ctx ['t'] = self._t
        ctx['episode'] = self._episode
        self._hist['contexts'].append(ctx)
        self._t += 1

    @property
    def history(self):
        return self._hist

    @property
    def contexts(self):
        """A list of contexts from the experiment."""
        return self._hist['contexts']

    @property
    def metadata(self):
        """Metadata from the experiment."""
        return self._hist['metadata']


class ExprimentalHistory(Callback):
    """Callback for recording an experiment's full history for multiple agents.

    Currently just records the contexts as in `History` along with the weights
    and traces for each agent, along with optional dict of additional values to
    compute from context.

    Attributes
    ----------
    history: dict
        The information recorded from the experiment.
    contexts: list[dict]
        A list of the contexts from each step of the experiment.
    metadata: dict
        A dictionary containing metadata about the experiment.
    records: list[list[dict]]
        A list of lists, one list per agent, recording information about the
        agent's state (e.g., weights & traces) from each time step of the
        experiment.


    Notes
    -------

    The history has the form::

        - metadata
            - start_time
            - end_time
            - num_episodes
            - max_steps
            - version
            - git_hash
            - total_time
            - environment
            - control policy
        - contexts
            - total_steps
            - episode
            - t (current timestep in episode)
            - obs
            - obs_p
            - a (action)
            - r (reward)
            - done
        - records (list of agent information)
            - episode
            - t (current timestep in episode)
            - weights
            - traces


    Remarks
    -------

    This is a compromise between the somewhat excessive `AgentHistory` (and
    having to record one history per-agent) and the more bare-bones `History`,
    where some information would be unavailable (traces, weights) or have to
    be recomputed (parameters).
    Here, everything (parameters, errors, etc.) is computable given the context,
    although recomputing everything might be expensive or tedious (with some
    stateful state-dependent paramters, for example a decaying stepsize).
    Could potentially collect information from each update_context as in
    `AgentHistory`, but I don't have a use for that now and it would make the
    code more complicated.
    """
    def __init__(self, agents=list(), compute={}):
        """
        Callback for recording an experiment's full history for multiple agents.
        See the class docstring for details.

        Parameters
        ----------
        agents: list
            List of learning agents to record information from.
        compute: dict
            Dictionary of functions for computing additional values from each
            context.
            The functions should accept a "context" (itself a dictionary with
            entries reflecting the current observation, action, reward, etc.)
            and return a value.
            Used to compute things like state-dependent parameter values,
            feature vectors, etc.
        """
        self._agents = agents
        self._compute = compute
        self._hist = {}
        self._hist['metadata'] = dict()
        self._hist['contexts'] = list()
        self._hist['records']  = [list() for i in agents]

    def on_experiment_begin(self, info=dict()):
        self._episode = None

        # Record some metadata
        self._hist['metadata'] = {
            'version'       : info['version'],
            'git_hash'      : info['git_hash'],
            'start_time'    : info['start_time'],
            'num_episodes'  : info['num_episodes'],
            'max_steps'     : info['max_steps'],
            'environment'   : info['environment'],
            'policy'        : info['policy'],
            'learners'      : info['learners'],
        }

    def on_experiment_end(self, info=dict()):
        self._hist['metadata']['num_episodes'] = self._episode + 1
        self._hist['metadata']['end_time'] = info['end_time']
        self._hist['metadata']['total_time'] = \
            (info['end_time'] - self._hist['metadata']['start_time']).total_seconds()

    def on_episode_begin(self, episode_ix, info=dict()):
        self._t = 0

        if self._episode is None:
            self._episode = 0
        else:
            self._episode += 1

    def on_step_begin(self, step_ix, info=dict()):
        # Information from agents (weights at time t, previous traces)
        for ix, agent in enumerate(self._agents):
            rec = {
                't': self._t,
                'episode': self._episode,
                'weights': agent.algo.weights,
                'traces' : agent.algo.traces,
            }
            self._hist['records'][ix].append(rec)

    def on_step_end(self, step_ix, info=dict()):
        # Context from environment
        ctx = {**info['context'], 't': self._t, 'episode': self._episode}

        for key, func in self._compute.items():
            ctx[key] = func(ctx)

        # Save the context information
        self._t += 1
        self._hist['contexts'].append(ctx)


    @property
    def history(self):
        return self._hist

    @property
    def contexts(self):
        """A list of contexts from the experiment."""
        return self._hist['contexts']

    @property
    def metadata(self):
        """Metadata from the experiment."""
        return self._hist['metadata']

    @property
    def records(self):
        """Records for each agent."""
        return self._hist['records']



# TODO: Improve this so it can track agents
# TODO: Improve this so it can compute arbitrary values using context
class History(Callback):
    """
    Records a history of the experiment.

    Attributes
    ----------
    history: dict
        The information recorded from the experiment.
    contexts: list[dict]
        A list of the contexts from each step of the experiment.
    metadata: dict
        A dictionary containing metadata about the experiment.


    Notes
    -----
    The history is stored as attributes of the callback, structured like:

        metadata
            - start_time
            - end_time
            - num_episodes
            - max_steps
            - version
            - git_hash
            - environment
            - policy
        contexts
            - total_steps
            - t (current timestep in episode)
            - episode
            - obs
            - obs_p
            - a (action)
            - r (reward)
            - done

    **Remarks**

    Serialization makes things difficult, but some things that are worth
    recording (such as environment and control policy) are hard to serialize.
    At this point I intend to use `pickle` to do this, which solves the problem
    of serialization but introduces potentially weird issues that I have yet to
    investigate.
    For example, does unpickling of an object fail catastrophically if the
    implementation of that object's class changes significantly?
    """
    def __init__(self):
        # Create data structure that will be filled by running the experiment
        self._hist = {}
        self._hist['metadata'] = dict()
        self._hist['contexts'] = list()

    def on_experiment_begin(self, info=dict()):
        self._episode = None
        self._hist['metadata'] = {
            'version'       : info['version'],
            'git_hash'      : info['git_hash'],
            'start_time'    : info['start_time'],
            'num_episodes'  : info['num_episodes'],
            'max_steps'     : info['max_steps'],
            'environment'   : info['environment'],
            'policy'        : info['policy'],
            'learners'      : info['learners'],
        }

    def on_experiment_end(self, info=dict()):
        self._hist['metadata']['num_episodes'] = self._episode + 1
        self._hist['metadata']['end_time'] = info['end_time']
        self._hist['metadata']['total_time'] = \
            (info['end_time'] - self._hist['metadata']['start_time']).total_seconds()

    def on_episode_begin(self, episode_ix, info=dict()):
        self._t = 0
        if self._episode is None:
            self._episode = 0
        else:
            self._episode += 1

    def on_episode_end(self, episode_ix, info=dict()):
        # TODO: Mark episodes where time ran out somehow?
        pass

    def on_step_begin(self, step_ix, info=dict()):
        pass

    def on_step_end(self, step_ix, info=dict()):
        ctx = {**info['context'], 't': self._t, 'episode': self._episode}
        self._t += 1
        self._hist['contexts'].append(ctx)

    def pretty_print(self):
        # Avoid showing warning on array scalar serialization
        json_tricks.NumpyEncoder.SHOW_SCALAR_WARNING=False
        print(json_tricks.dumps(self._hist, indent=2))
        json_tricks.NumpyEncoder.SHOW_SCALAR_WARNING=True

    @property
    def history(self):
        return self._hist

    @property
    def contexts(self):
        """A list of contexts from the experiment."""
        return self._hist['contexts']

    @property
    def metadata(self):
        """Metadata from the experiment."""
        return self._hist['metadata']


class Progress(Callback):
    """Progress display callback."""
    import sys
    def __init__(self, stream=None):
        if stream is None:
            stream = self.sys.stdout
        self.stream = stream

    def on_experiment_begin(self, info=dict()):
        self.num_episodes = info['num_episodes']
        self.max_steps = info['max_steps']
        self.cumulative_steps = 0
        self.prev_total_steps = 0
        # Handle leaving number of episodes unspecified
        if self.num_episodes is None:
            self.fmt_string = ("Episode {episode_ix} "
                               "(total steps: {total_steps:d}, last {last_steps})")
        else:
            self.fmt_string = ("Episode {episode_ix} of {num_episodes} "
                               "(total steps: {total_steps:d}, last {last_steps})")

    def on_episode_end(self, episode_ix, info):
        total_steps = info['total_steps']
        self.episode_steps = total_steps - self.prev_total_steps
        self.prev_total_steps = total_steps
        msg = self.fmt_string.format(
            episode_ix=episode_ix+1,
            num_episodes=self.num_episodes,
            total_steps=total_steps,
            last_steps=self.episode_steps
        )
        # Print messages
        print(msg, file=self.stream, flush=True, end="\r")

    # def on_step_end(self, step_ix, info=dict()):
    #     self.cumulative_steps += 1
    #     msg = "Episode %d of %d, step: %d (cumulative steps: %d)"%(
    #         self.episode_ix, self.num_episodes, step_ix,
    #         self.cumulative_steps)
    #     # Print messages
    #     print(msg, file=self.stream, flush=True, end="\r")


    def on_experiment_end(self, info=dict()):
        print("\n", end="", file=self.stream, flush=True)


class VerboseProgress(Callback):
    """More detailed progress display callback."""
    import sys
    def __init__(self, stream=None):
        if stream is None:
            stream = self.sys.stdout
        self.stream = stream

    def on_experiment_begin(self, info=dict()):
        self.num_episodes = info['num_episodes']
        self.max_steps = info['max_steps']
        self.cumulative_steps = 0

    def on_episode_begin(self, episode_ix, info=dict()):
        msg = "Episode %d of %d (total steps: %d)"%(
            episode_ix+1, self.num_episodes, info['total_steps'])
        print(msg, end="\r", file=self.stream, flush=True)

    def on_episode_end(self, episode_ix, info):
        total_steps = info['total_steps']
        msg = "Episode %d of %d (total steps: %d)"%(
            episode_ix, self.num_episodes, total_steps)
        # Print messages
        print(msg, file=self.stream, flush=True)
        print("Episode steps: %d"%(total_steps - self.cumulative_steps))
        # Ready for next episode
        self.cumulative_steps = total_steps


    def on_experiment_end(self, info=dict()):
        print()


class RemoteMonitor(Callback):
    """Stream information about the experiment to a socket."""
    pass


class StreamMonitor(Callback):
    """Write information about the experiment to a stream."""
    pass
