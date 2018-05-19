import numpy as np
from .feature_base import Feature


class BoyanFeatures(Feature):
    """Features for the Boyan Chain.

    [See page 6 of the iLSTD paper](https://papers.nips.cc/paper/3092-ilstd-eligibility-traces-and-convergence-analysis.pdf)
    for an example using the chain with these features.
    """
    __mapping = {
         0 : np.array([0.0 , 0.0 , 0.0 , 1.0 ]),
         1 : np.array([0.0 , 0.0 , 0.25, 0.75]),
         2 : np.array([0.0 , 0.0 , 0.5 , 0.5 ]),
         3 : np.array([0.0 , 0.0 , 0.75, 0.25]),
         4 : np.array([0.0 , 0.0 , 1.0 , 0.0 ]),
         5 : np.array([0.0 , 0.25, 0.75, 0.0 ]),
         6 : np.array([0.0 , 0.5 , 0.5 , 0.0 ]),
         7 : np.array([0.0 , 0.75, 0.25, 0.0 ]),
         8 : np.array([0.0 , 1.0 , 0.0 , 0.0 ]),
         9 : np.array([0.25, 0.75, 0.0 , 0.0 ]),
        10 : np.array([0.5 , 0.5 , 0.0 , 0.0 ]),
        11 : np.array([0.75, 0.25, 0.0 , 0.0 ]),
        12 : np.array([1.0 , 0.0 , 0.0 , 0.0 ]),
    }
    def __init__(self, *args, **kwargs):
        self.mapping = {k: v.copy() for k, v in self.__mapping.items()}

    def __call__(self, obs):
        return self.mapping[obs]

    def __len__(self):
        return 4

    def as_matrix(self):
        num_states = len(self.mapping)
        ret = np.empty((num_states, len(self)))
        for i in range(num_states):
            ret[i] = self.mapping[i]
        return ret
