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
