"""Base class for learning algorithms."""
import abc
import inspect


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

        print(name) # TODO: Remove
        return super(LearningAlgorithmMeta, meta).__new__(meta, name, parents, attrs)

    @abc.abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def learn(*args, **kwargs):
        pass

    @abc.abstractmethod
    def update(self, params):
        """Update from new experience, using the supplied parameters which are
        then fed to `learn` according to the method signature.


        Parameters
        ----------
        params: dict
            A dictionary containing the parameters to pass to `learn`.
        """
        pass

    @abc.abstractmethod
    def reset(self):
        pass


class LearningAlgorithm(metaclass=LearningAlgorithmMeta):
    """Learning algorithm base class."""
    def update(self, context):
        """Update from new experience, using the supplied parameters which are
        then fed to `learn` according to `_learn_params`, which is determined
        at class creation.

        Parameters
        ----------
        context: dict
            A dictionary containing all information needed by `self.learn`.
        """
        # Extract parameters to feed to `self.learn` from `params`
        args = [context[key] for key in self._learn_params]
        return self.learn(*args)
