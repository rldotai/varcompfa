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
    params: dict, optional
        A dictionary of parameters, of the form `param_name: <value>`, where
        `<value>` can be either a constant (e.g., a float) or a callable that
        accepts a context.
    reward_func: callable, optional
        An reward function accepts a context and returns a `float`.
        Modifying the reward function can be useful in some cases, e.g. for
        reward shaping or for predicting other quantities than the return.
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
    def __init__(self, algo, phi, params=dict(), reward_func=None, metadata=dict()):
        self.algo = algo
        self.phi = phi
        self.params = params
        self.metadata = metadata
        # Override default reward function if an alternative is provided
        if reward_func is not None:
            self.reward_func = reward_func

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

    def reward_func(self, context: dict) -> float:
        """The reward function, which by default does nothing unless overriden
        during initialization.

        Parameters
        ----------
        context: dict
            A basic context containing information about a single timestep

        Returns
        -------
        reward: float
            The reward derived from the given context.
        """
        return context['r']

    def eval_params(self, context: dict):
        """Evaluate the parameter functions for the supplied context."""
        return {key: val(context) if callable(val) else val
                        for key, val in self.params.items()}

    def update(self, context: dict, check_conflict=True):
        """Update the learning agent from the current context (e.g., the
        information available at the timestep).

        Parameters
        ----------
        context: dict
            A dictionary containing information about the current timestep.
            The agent will then compute the feature representation and
            context-dependent parameters to be used when updating the agent.
        check_conflict: bool (default True)
            If true, check if parameters passed via `context` have a name
            conflict with those computed as part of the update.

        Notes
        -----
        The parameters passed by `context` take precedence over the parameters
        computed as part of this function (including the feature vectors).
        By default, when this occurs a warning will be printed, but sometimes
        it is necessary/convenient to override the computed parameter values.
        """
        # Check if we clobber anything in `context` with `params`
        if check_conflict:
            _intersect = set(self.params).intersection(context)
            if _intersect:
                _logger.warn("agent.update(): parameter name conflict: %s"%_intersect)

        # Compute the reward for the given context
        ctx = {
            **context,
            'r': self.reward_func(context),
        }

        # Compute parameters given the current context
        _params = self.eval_params(ctx)

        # Create the combined context
        ctx = {
            'x': self.phi(ctx['obs']),
            'xp': self.phi(ctx['obs_p']),
            **_params,
            **ctx
        }
        ctx['update_result'] = self.algo.update(ctx)
        return ctx

    def terminal_context(self, defaults=dict()):
        """Return a suitable context for terminal states, overriding the
        context provided by `defaults`.
        This entails setting `done` to `True`, and returning a zero-vector
        for the features of the current state and its successor.
        """
        ctx = {
            **defaults,
            'xp' : np.zeros(len(self.phi)),
            'done': True,
            'r' : 0,
        }
        _params = self.eval_params(ctx)
        ctx = {**ctx, **_params}
        return ctx

    def get_td_error(self, context):
        """Compute the TD-error at a given step."""
        params  = self.eval_params(context)
        vx      = self.get_value(context['obs'])
        vx_p    = self.get_value(context['obs_p'])
        delta   = context['r'] + params['gm_p']*vx_p - vx
        return delta

    def get_value(self, obs: np.ndarray):
        """Get the value assigned to the current observation by the learning
        algorithm under the agent's function approximation scheme.
        """
        return self.algo.get_value(self.phi(obs))

    def start_episode(self):
        """Get ready to start a new episode."""
        self.algo.start_episode()

    def get_config(self):
        # TODO: Finish this, or eliminate it if unnecessary
        raise NotImplementedError()

