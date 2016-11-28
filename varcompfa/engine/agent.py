"""
Implementation of agent wrapper classes
"""
import numpy as np


class Agent:
    """Agent class, for encapsulating a learning algorithm, its function
    approximator, and the possibly state-dependent parameters for updating it.
    """
    def __init__(self, algo, phi, param_funcs=dict()):
        self.algo = algo
        self.phi = phi
        self.param_funcs = param_funcs

    def update(self, context):
        """Update the learning agent from the current context (e.g., the
        information available at the timestep).
        """
        # Compute features from context
        context['x'] = self.phi(context['obs'])
        context['xp'] = self.phi(context['obs_p'])
        # Check if we clobber anything in `context` with param_funcs
        _intersect = set(self.param_funcs).intersection(context)
        if _intersect:
            logger.warn("agent.update(): parameter name conflict: %s"%_intersect)

        # Compute parameters given the current context
        _params = {key: func(context) for key, func in self.param_funcs.items()}
        _ctx = {**_params, **context}
        # print(_ctx) # TODO: REMOVE
        ret = self.algo.update(_ctx)
        return ret

    def terminal_update(self, context):
        """Perform update in the terminal state.

        TODO: Unneeded? Maybe enforcing termination conditions can be done
        separately.
        """
        context['x'] = self.phi(context['obs'])
        context['xp'] = np.zeros_like(context['x'])
        _params = {key: func(context) for key, func in self.param_funcs.items()}
        _ctx = {**_params, **context}
        # print(_ctx)
        ret = self.algo.update(_ctx)
        return ret


    def act(self, obs):
        """Select an action according to the current observation using the
        learning algorithm (`self.algo`).
        """
        x = self.phi(obs)
        return self.algo.act(x)

    def get_value(self, obs):
        """Get the value assigned to the current observation by the learning
        algorithm under the agent's function approximation scheme.
        """
        return self.algo.get_value(self.phi(obs))

    def get_params(self, context):
        """Evaluate the parameter functions for the supplied context."""
        return {key: func(context) for key, func in self.param_funcs.items()}

    def save(self):
        pass

    def load(self):
        pass

    def __str__(self):
        return super().__str__()
