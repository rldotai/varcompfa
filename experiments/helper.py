"""Helper functions for handling things common to various experiments."""
import os
import json_tricks as jt
import varcompfa as vcf
import config

# Logging
import logging
logger = logging.getLogger(__name__)


def find_controller(name, env_name=''):
    """Find a controller, checking first if `name` is a file and if that fails
    searching in `varcompfa/controllers/<env_name>` for a controller named
    appropriately.
    """
    logger.debug('Searching for controller named: %s'%name)
    if os.path.isfile(name):
        return os.path.abspath(name)

    # Try searching for the controller's path in the appropriate directory
    control_dir = os.path.join(config.CONTROLLER_DIR, env_name)
    if os.path.isfile(os.path.join(control_dir, name)):
        return os.path.join(control_dir, name)
    elif os.path.isfile(os.path.join(control_dir, name) + '.json'):
        return os.path.join(control_dir, name) + '.json'
    else:
        raise Exception('Cannot find controller named: %s for env: %s'%(
            name, env_name))

def load_controller(name, env_name):
    """Load a JSON-serialized controller."""
    control_path = find_controller(name, env_name)
    logger.debug('Loading controller from: %s'%control_path)
    return jt.load(open(control_path, 'r'))
