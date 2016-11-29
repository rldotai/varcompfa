"""
Callbacks to execute at various points in the course of running an experiment.

Based on Keras' callbacks.
"""
import time


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
