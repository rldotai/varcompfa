"""
Implementation of agent wrapper classes
"""
import numpy as np

# TODO: Set up logging for the whole package
import logging
_logger = logging.getLogger()


class Agent:
    """Agent class, for encapsulating a learning algorithm, its function
    approximator, and the possibly state-dependent parameters for updating it.


    Parameters
    ----------
    algo : varcompfa.algos.LearningAgent
        The learning algorithm that the Agent wraps.
        It must have an `update` method capable of handling a dictionary
        containing the information needed to perform updates.
    phi : callable
        A function that maps observations to features used by the learning algo.
    params: dict
        A dictionary of parameters, of the form `param_name: <value>`, where
        `<value>` can be either a constant (e.g., a float) or a callable that
        accepts a context.
    metadata: dict, optional
        An optional dictionary for adding metadata to the agent, e.g. for
        annotations on the experiment it was trained on.

    Note
    ----
    We make use of contexts to make it possible to compute things without
    knowing too much in advance about what we're computing.
    Taken to extremes, this approach would devolve into making everything a
    global variable, which is probably not a good idea.
    However, since this codebase is oriented towards reinforcement learning,
    we can restrict what the context exposes.

    The baseline context is the state, action, reward, and next state.
    From this, we can compute the features for the state and its successor,
    and add those to the context.
    If the parameters are functions (e.g., state-dependent), we compute those
    as well and include them, passing them to the learning algorithm.
    At this point, everything necessary for the learning algorithm to perform
    an update should be available.
    """
    def __init__(self, algo, phi, params=dict(), metadata=dict()):
        self.algo = algo
        self.phi = phi
        self.params = params
        self.metadata = metadata

    def update(self, context: dict):
        """Update the learning agent from the current context (e.g., the
        information available at the timestep).
        """
        # Check if we clobber anything in `context` with `params`
        _intersect = set(self.params).intersection(context)
        if _intersect:
            _logger.warn("agent.update(): parameter name conflict: %s"%_intersect)

        # Compute parameters given the current context
        _params = {key: val(context) if callable(val) else val
                        for key, val in self.params.items()}
        ctx = {
            'x': self.phi(context['obs']),
            'xp': self.phi(context['obs_p']),
            **_params,
            **context
        }
        res = self.algo.update(ctx)
        # TODO: Either return all the results or remove?
        # ctx['result'] = res
        # return ctx
        return res

    def act(self, obs: np.ndarray):
        """Select an action according to the current observation using the
        learning algorithm (`self.algo`).

        Parameters
        ----------
        obs : numpy.ndarray
            The observation that the agent uses to determine the action to take.

        Returns
        -------
        action: numpy.ndarray
            The action selected by the algorithm given the features `phi(x)`.
            (It is an array because `json_tricks` does not presently handle
            serializing non-array instances of numpy datatypes.
        """
        x = self.phi(obs)
        return np.array(self.algo.act(x))

    def get_value(self, obs: np.ndarray):
        """Get the value assigned to the current observation by the learning
        algorithm under the agent's function approximation scheme.
        """
        return self.algo.get_value(self.phi(obs))

    def eval_params(self, context: dict):
        """Evaluate the parameter functions for the supplied context."""
        return {key: val(context) if callable(val) else val
                        for key, val in self.params.items()}

    def get_config(self):
        # TODO: Finish this, or eliminate it if unnecessary
        raise NotImplementedError()

