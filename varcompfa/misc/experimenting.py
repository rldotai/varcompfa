import itertools
import os
import pickle


def parameter_search(base, vary):
    names, values = zip(*vary.items())
    for case in itertools.product(*values):
        yield {**base, **dict(zip(names, case))}

