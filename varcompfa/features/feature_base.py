"""
Base classes for features.
"""
import abc
import numpy as np
from varcompfa.utils import get_class_string, load_class


class Feature(metaclass=abc.ABCMeta):
    """Abstract base class for feature functions."""
    @abc.abstractmethod
    def __call__(self, *args):
        """The operation mapping an input to its representation under the feature."""
        pass

    @abc.abstractmethod
    def __len__(self):
        """Length of the output vector produced by the feature for a single observation."""
        pass

    def __str__(self):
        """String representation of the feature's specification."""
        pass

    @abc.abstractmethod
    def get_config(self):
        """Get the configuration necessary to full specify a feature."""
        pass

    @abc.abstractclassmethod
    def from_config(cls, config):
        """Instantiate the feature from a configuration object."""
        pass

    @staticmethod
    def from_dict(dct):
        """Load a feature from a `dict` of the form returned by `to_dict`."""
        class_name = dct['class_name']
        config = dct['config']
        cls = load_class(class_name)
        return cls.from_config(config)

    def to_dict(self):
        """Convert a feature to a dictionary which fully specifies it, which
        should provide all the information needed to preserve/instantiate it.

        This has to act recursively, since some features may be composed of
        other child features.
        """
        # TODO




def load_feature(class_name, config):
    pass

