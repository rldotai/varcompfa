"""Base class for learning algorithms."""
import abc
import inspect
from varcompfa.utils import get_class_string, load_class


class LearningAlgorithmMeta(type, metaclass=abc.ABCMeta):
    """Abstract base class for learning algorithms"""

    def __new__(meta, name, parents, attrs):
        # Get the signature only for classes derived from the base class.
        if name is not 'LearningAlgorithm':
            # Get the signature of the `learn` method, and the parameter ordering
            learn_signature = inspect.signature(attrs['learn'])
            learn_params = [i for i in learn_signature.parameters.keys() if i is not 'self']
            attrs['_learn_params'] = tuple(learn_params)
            # More complicated setup might be required if we start incorporating
            # keyword parameters or ones that are optional

        return super(LearningAlgorithmMeta, meta).__new__(meta, name, parents, attrs)

    @abc.abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def get_value(self, *args, **kwargs):
        """Compute the value for the supplied features."""
        pass

    @abc.abstractmethod
    def learn(*args, **kwargs):
        pass

    @abc.abstractmethod
    def reset(self):
        pass

    @abc.abstractmethod
    def start_episode(self):
        """Perform any actions (eg, clear traces) for starting a new episode."""
        pass

    @abc.abstractmethod
    def get_config(self):
        """Get the configuration for the algorithm, i.e., all information that
        would be needed to instantiate it, as a `dict`.
        """
        pass

    @abc.abstractclassmethod
    def from_config(cls, config):
        """Load the algorithm from a configuration stored in a dict."""
        pass


class LearningAlgorithm(metaclass=LearningAlgorithmMeta):
    """Learning algorithm base class."""
    def update(self, context):
        """Update from new experience.
        Uses the supplied parameters from `context` which are then fed to the
        agent's `learn` function according to `_learn_params`.

        Via some metaclass programming, we determine the signature of `learn`
        at class creation, and define `_learn_params` accordingly.


        Parameters
        ----------
        context: dict
            A dictionary containing all information needed by `self.learn`.


        Returns
        -------
        update_result:
            The value returned by the algorithm's `learn()` method.
        """
        # Extract parameters to feed to `self.learn` from `params`
        args = [context[key] for key in self._learn_params]
        return self.learn(*args)

    def to_dict(self):
        """Get the algorithm's class string and its configuration, which
        should provide all the information necessary to preserve/instantiate it.
        """
        cfg = {
            'class_name': get_class_string(self),
            'config': self.get_config()
        }
        return cfg

    @staticmethod
    def from_dict(dct):
        """Load an algorithm from a `dict` of the form returned by `to_dict`"""
        class_name = dct['class_name']
        config = dct['config']
        cls = load_class(class_name)
        return cls.from_config(config)


def load_algorithm(class_name, config):
    """Load an algorithm from a configuration.

    The configuration should be of the sort returned by an algorithm's
    `get_config` method, which is defined generically by the parent class
    `LearningAlgorithm`.
    """
    cls = load_class(class_name)
    return cls.from_config(config)
