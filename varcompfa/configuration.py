"""Configuration for the module (e.g., setting up logging).

See: `openai/gym/gym/configuration.py` for the inspiration, but note that they
elect to configure the root logger, which is a bit of an odd decision.
"""
import logging
import sys

import varcompfa as vcf

# -----------------------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)
root_logger = logging.getLogger()

# Should be 'varcompfa'
package_name = vcf.__name__
vcf_logger = logging.getLogger(package_name)

# Set up log message formatting
basic_formatter = logging.Formatter("[%(asctime)s] %(message)s")
verbose_formatter = logging.Formatter(
    "[%(levelname)s: %(name)s: %(asctime)s] %(message)s"
)
detailed_formatter = logging.Formatter(
    "%(name)s:%(levelname)s %(module)s:%(lineno)d:  %(message)s"
)
# formatter = basic_formatter
formatter = verbose_formatter

# Set up handler
handler = logging.StreamHandler(sys.stderr)
handler.setFormatter(formatter)


def logger_setup(level=logging.DEBUG):
    vcf_logger.addHandler(handler)
    vcf_logger.setLevel(level)
    # Avoid duplicate logging thanks to OpenAI Gym
    vcf_logger.propagate = False


def undo_logger_setup():
    vcf_logger.removeHandler(handler)
    vcf_logger.setLevel(logging.NOTSET)
    vcf_logger.propagate = True
