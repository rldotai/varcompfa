"""
Callbacks to execute at various points in the course of running an experiment.

Based on Keras' callbacks.
"""
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

    def _set_params(self, params):
        self.params = params

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


class History(Callback):
    """

    Records a history of the experiment, of the form:

    - start_time
    - end_time
    - num_episodes
    - max_steps
    - version
    - git_hash
    - episodes
        - steps (a list of contexts-- obs, action, next obs, reward, done)
        - update_results (a list of update results returned by the learning algos)
    """
    def __init__(self):
        # Create data structure that will be filled by running the experiment
        self._hist = {}

    def on_experiment_begin(self, info=dict()):
        # TODO: Compute some of these in the PolicyEval class?
        self._hist['version'] = info['version']
        self._hist['git_hash'] = info['git_hash']
        self._hist['start_time'] = info['start_time']
        self._hist['num_episodes'] = info['num_episodes']
        self._hist['max_steps'] = info['max_steps']
        # Set up list of episodes
        self._hist['episodes'] = list()

    def on_experiment_end(self, info=dict()):
        # TODO: Compute in the PolicyEval class?
        self._hist['end_time'] = time.time()

    def on_episode_begin(self, episode_ix, info=dict()):
        self.episode = {
            'contexts': list(),
            'update_results': list(),
        }

    def on_episode_end(self, episode_ix, info=dict()):
        self._hist['episodes'].append(self.episode)

    def on_step_begin(self, step_ix, info=dict()):
        pass

    def on_step_end(self, step_ix, info=dict()):
        self.episode['contexts'].append(info['context'])
        self.episode['update_results'].append([i for i in info['update_results']])

    def pretty_print(self):
        # Avoid showing warning on array scalar serialization
        json_tricks.NumpyEncoder.SHOW_SCALAR_WARNING=False
        print(json_tricks.dumps(self._hist, indent=2))
        json_tricks.NumpyEncoder.SHOW_SCALAR_WARNING=True

    @property
    def history(self):
        return self._hist

    def flat_contexts(self):
        """Return a flattened list of all steps in an episode."""
        episodes = self.history['episodes']
        return [i for ep in episodes for i in ep['contexts']]

    def flat_updates(self):
        """Return a flattened list of all update results."""
        episodes = self.history['episodes']
        return [i for ep in episodes for i in ep['update_results']]


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

    def on_episode_begin(self, episode_ix, info=dict()):
        msg = "Episode %d of %d (total steps: %d)"%(
            episode_ix, self.num_episodes, info['total_steps'])
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
