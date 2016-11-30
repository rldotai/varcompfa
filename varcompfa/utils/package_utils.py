"""
Utilities relating to the package itself, for example, finding the current
version or the repo's git hash.
"""
import inspect
import os
import subprocess
import varcompfa as vcf


def current_git_hash():
    """Returns the hash of the current git branch.

    Useful for identifying the state of the codebase when running experiments.
    """
    # Find the package's directory and its parent (the repo directory)
    package_dir = os.path.dirname(inspect.getfile(vcf))
    parent_dir = os.path.dirname(package_dir)
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=parent_dir)
    ret = sha.decode('ascii').strip()
    return ret

def current_version():
    """Returns the current version of the `varcompfa` package."""
    return vcf.__version__
