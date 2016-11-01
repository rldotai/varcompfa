"""
Base classes for features.
"""
import abc 
import numpy as np 


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

    @property
    @abc.abstractmethod
    def params(self):
        """The parameters necessary to fully specify the feature."""
        pass
   
    def __str__(self):
        """String representation of the feature's specification."""
        return str(self.params)
