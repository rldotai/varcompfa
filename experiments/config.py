"""
Configuration for experiments.
"""
import logging
import os
import sys

# Configure logging
logger = logging.getLogger(__name__)
# TODO: Setting root logger level is probably not smart...
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)

# Set up log message formatting
basic_formatter = logging.Formatter('[%(asctime)s] %(message)s')
verbose_formatter = logging.Formatter('[%(levelname)s: %(name)s: %(asctime)s] %(message)s')
detailed_formatter = logging.Formatter('%(name)s:%(levelname)s %(module)s:%(lineno)d:  %(message)s')
# formatter = basic_formatter
formatter = verbose_formatter

# Set up handler
handler = logging.StreamHandler(sys.stderr)
handler.setFormatter(formatter)

# Common directories
EXPERIMENT_DIR  = os.path.dirname(os.path.realpath(__file__))
PROJECT_DIR     = os.path.dirname(EXPERIMENT_DIR)
CONTROLLER_DIR  = os.path.join(PROJECT_DIR, 'controllers')
RESULT_DIR      = os.path.join(PROJECT_DIR, 'results')
PLOT_DIR        = os.path.join(PROJECT_DIR, 'plots')
