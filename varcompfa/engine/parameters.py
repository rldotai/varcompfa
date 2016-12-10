"""Code for state-dependent parameter functions."""


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

class ExponentialDecay:
    """A parameter that decays exponentially with the total number of steps.

    That is, for `t = total_steps`,
        `value(t) = initial_value * e^(-decay_rate * total_steps)`
    """
    from math import exp
    def __init__(self, initial_val: float, decay_rate: float, terminal_value=None):
        self.initial_value = initial_value
        self.decay_rate = decay_rate

    def __call__(self, context):
        if context['done']:
            return self.terminal_value
        return self.initial_value * self.exp(-decay_rate*context['total_steps'])

class PowerDecay:
    """A parameter that decays according to a power law with respect to the
    total number of steps.

    That is, for `t = total_steps`,
        `value(t) = base * (t)**(-exponent)`

    """
    def __init__(self, base: float, exponent: float, terminal_value=None):
        assert(exponent > 0)
        self.base = base
        self.exponent = exponent

    def __call__(self, context):
        if context['done']:
            return self.terminal_value
        return self.base * context['total_steps']**(-self.exponent)
