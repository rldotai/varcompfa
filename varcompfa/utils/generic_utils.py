"""
Generic utilities
"""
import importlib
import inspect
import pydoc


def find_module(obj):
    """Find the name of the module where `obj` is defined.

    For example, `vcf.TD` becomes `vcf.algos.td`
    """
    if inspect.isclass(obj):
        return inspect.getmodule(obj).__name__
    else:
        return inspect.getmodule(obj.__class__).__name__

def get_class_string(obj):
    """Get the class string for an object

    For example, `vcf.TD` becomes `vcf.algos.td.TD`, as does an instantiation
    of that class.

    May be slightly dicey because Python's object model doesn't always permit
    you to find the right thing.
    For example, a `numpy` array returns `numpy.ndarray`, which is the correct
    type, but attempting to instantiate it the same way you would create an
    array via `numpy.array([1,2])` will yield a zero-valued array with shape
    `(1,2)`, rather than an array of shape `(2,)` with values `[1,2]`.
    """
    module_name = find_module(obj)
    if inspect.isclass(obj):
        return module_name + '.' + obj.__name__
    else:
        return module_name + '.' + obj.__class__.__name__

def load_class(class_string):
    """Load a class from a class string.

    Parameters
    ----------
    class_string: str
        The canonical class string, e.g. the result of `str(obj.__class__)`,
        where `obj` is an instantiation of a Python class.

    For example, `class_string` could be `"varcompfa.algos.TD.td"`

    The `pydoc` package implements this nicely via `locate`, although we could
    do something similar via:

    ```
    class_data = class_string.split('.')
    module_path = '.'.join(class_data[:-1])
    class_name = class_data[-1]
    module = importlib.import_module(module_path)
    return getattr(module, class_name)
    ```
    """
    cls = pydoc.locate(class_string)
    if not isinstance(cls, type):
        raise TypeError("Non-class object loaded from: %s"%class_string)
    return cls

