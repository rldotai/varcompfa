"""Code for state-dependent parameter functions."""
#TODO: Add __str__ and __repr__ methods

# Logging
import logging
logger = logging.getLogger(__name__)


class Constant:
    """A constant parameter, which has the option to return a different value
    if the terminal state has been reached.
    """
    def __init__(self, value, terminal_value=None):
        if terminal_value is None:
            terminal_value = value
        self.value = value
        self.terminal_value = terminal_value

    def __call__(self, context):
        if context['done']:
            return self.terminal_value
        return self.value

    def __str__(self):
        return 'Constant(%s, %s)'%(self.value, self.terminal_value)

# TODO: Write this
# TODO: Parameter algebra?
# A parameter that comes as a composition of other parameters
class Composed:
    pass

class Map:
    """A parameter that maps observations to parameter values using a
    dictionary or other object that has a `.get` method.
    """
    def __init__(self, mapping, key='obs', default=None):
        self.mapping = mapping
        self.default = default
        self.key     = key

    def __call__(self, context):
        ret = self.mapping.get(context[self.key], self.default)
        if ret is not None:
            return ret
        else:
            raise Exception("No value specified for obs: %s"%(context['obs']))

    def __str__(self):
        return "Map(%s, default=%s)"%(self.mapping, self.default)

###############################################################################
# Stepsize Parameters
###############################################################################
class EpisodicExponential:
    """A parameter that decays exponentially with the number of episodes.

    That is, for `t = episode_count`,
        `value(t) = initial_value * e^(-decay_rate * t)`

    NOTE
    ----
    By successfully completed episodes, we mean that the `episode_count` only
    increments when the environment reaches a terminal state, not when
    `max_steps` is reached.
    That is, `episode_count` is creased by one upon receiving `done=True` in
    the context passed to the parameter.
    """
    from math import exp
    def __init__(self, initial_val: float, decay_rate: float, terminal_value=None):
        self.initial_value = initial_value
        self.decay_rate = decay_rate
        self.terminal_value = terminal_value
        # Set episode_count and value
        self.episode_count = 0
        self._value = initial_value*self.exp(-decay_rate * self.episode_count)

    def __call__(self, context):
        if context['done']:
            self._value = self.initial_value*self.exp(-self.decay_rate * self.episode_count)
            if self.terminal_value:
                return self.terminal_value
        return self._value


class EpisodicPowerLaw:
    """A parameter that decays according to a power law with respect to the
    number of completed episodes.

    That is, for `t = episode_count`,
        `value(t) = base * (t+1)**(-exponent)`

    NOTE
    ----
    By successfully completed episodes, we mean that the `episode_count` only
    increments when the environment reaches a terminal state, not when
    `max_steps` is reached.
    That is, `episode_count` is creased by one upon receiving `done=True` in
    the context passed to the parameter.
    """
    def __init__(self, base: float, exponent: float, terminal_value=None):
        assert(exponent > 0)
        self.base = base
        self.exponent = exponent
        self.terminal_value = terminal_value
        # Set episode_count and value
        self.episode_count = 0
        self._value = self.base * (1 + self.episode_count)**(-self.exponent)

    def __call__(self, context):
        if context['done']:
            self.episode_count +=1
            self._value = self.base * (1 + self.episode_count)**(-self.exponent)
            if self.terminal_value:
                return self.terminal_value
        return self._value


class EpisodicParameter:
    """A parameter set according to an arbitrary single-argument function,
    with the argument in question being the number of successfully completed
    episodes.

    NOTE
    ----
    The number of episodes starts at zero, so the initial value is
    `value(0) = func(0)`, where `func` is the provided function.
    By successfully completed episodes, we mean that the `episode_count` only
    increments when the environment reaches a terminal state, not when
    `max_steps` is reached.
    That is, `episode_count` is creased by one upon receiving `done=True` in
    the context passed to the parameter.
    """
    def __init__(self, func, terminal_value=None):
        self.func = func
        self.terminal_value = terminal_value
        self.episode_count = 0
        self._value = self.func(self.episode_count)

    def __call__(self, context):
        if context['done']:
            self.episode_count +=1
            self._value = self.func(self.episode_count)
            if self.terminal_value:
                return self.terminal_value
        return self._value

class StepwiseParameter:
    """A parameter set according to an arbitrary single-argument function,
    with the argument in question being the total number of steps.
    """
    def __init__(self, func):
        self.func = func
        self.terminal_value = terminal_value

    def __call__(self, context):
        if context['done'] and self.terminal_value:
            return self.terminal_value
        return self.func(context['total_steps'])

class StepwiseExponential:
    """A parameter that decays exponentially with the total number of steps.

    That is, for `t = total_steps`,
        `value(t) = initial_value * e^(-decay_rate * t)`
    """
    from math import exp
    def __init__(self, initial_val: float, decay_rate: float, terminal_value=None):
        self.initial_value = initial_value
        self.decay_rate = decay_rate

    def __call__(self, context):
        if context['done'] and self.terminal_value:
            return self.terminal_value
        return self.initial_value * self.exp(-decay_rate*context['total_steps'])

class StepwisePowerLaw:
    """A parameter that decays according to a power law with respect to the
    total number of steps.

    That is, for `t = total_steps`,
        `value(t) = base * (t+1)**(-exponent)`
    """
    def __init__(self, base: float, exponent: float, terminal_value=None):
        assert(exponent > 0)
        self.base = base
        self.exponent = exponent

    def __call__(self, context):
        if context['done'] and self.terminal_value:
            return self.terminal_value
        return self.base * (1 + context['total_steps'])**(-self.exponent)
