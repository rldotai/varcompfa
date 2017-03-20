import itertools
import os
import pickle


def parameter_search(base, vary):
    names, values = zip(*vary.items())
    for case in itertools.product(*values):
        yield {**base, **dict(zip(names, case))}

def save_agent(agent, path, overwrite=False):
    # Handle path being a path or file-like object
    # If it's a path, check if it's got an extension already else append `pkl`
    if not isinstance(agent, vcf.Agent):
        raise Exception("This function is for saving agents...")
    if os.path.exists(path) and not overwrite:
        raise Exception("File exists at: %s" % (path,))
    pickle.dump(agent, path, protocol=4)

def load_agent(path):
    # Handle path being a path or file-like object
    # If it's a path, check if it's got an extension already,
    # otherise try after appending `pkl`
    agent = pickle.load(path, protocol=4)
    if not isinstance(agent, vcf.Agent):
        raise Exception("This function is for loading agents...")
    return agent
