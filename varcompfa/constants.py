"""
Important constant values/types that other parts of the codebase rely upon.
"""

class Singleton(type):
    def __new__(meta, name, parents, attrs):
        if any(isinstance(cls, meta) for cls in parents):
            raise TypeError("Cannot inherit from singleton class")
        attrs['_instance'] = None
        return super(Singleton, meta).__new__(meta, name, parents, attrs)
