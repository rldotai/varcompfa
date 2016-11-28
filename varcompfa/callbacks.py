"""
Callbacks to execute at various points in the course of running an experiment.

Based on Keras' callbacks.
"""


class Callback:
    """Base class used to build new callbacks."""
    def __init__(self):
        pass

    def _set_params(self, params):
        self.params = params

    def on_experiment_begin(self, logs=dict()):
        pass

    def on_experiment_end(self, logs=dict()):
        pass

    def on_episode_begin(self, episode_ix, logs=dict()):
        pass

    def on_episode_end(self, episode_ix, logs=dict()):
        pass

    def on_step_begin(self, step_ix, logs=dict()):
        pass

    def on_step_end(self, step_ix, logs=dict()):
        pass


class ExampleCallback(Callback):
    def on_train_begin(self):
        print("Started training")

    def on_train_end(self):
        print("End of training")

    def on_episode_begin(self, episode_ix):
        print("Started episode: %d"%episode_ix)

    def on_episode_end(self, episode_ix):
        print("End of episode: %d"%episode_ix)

    def on_step_begin(self, step_ix):
        print("Begin step: %d"%step_ix)

    def on_step_end(self, step_ix):
        print("End step: %d"%step_ix)
