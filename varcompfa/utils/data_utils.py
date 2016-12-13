"""Utilities for working with data, e.g., CSV files, JSON files, and so on."""
import json_tricks

def dump_json(obj, destpath, *args, **kwargs):
    json_tricks.NumpyEncoder.SHOW_SCALAR_WARNING=False
    ret = json_tricks.dump(obj, destpath, *args, **kwargs)
    json_tricks.NumpyEncoder.SHOW_SCALAR_WARNING=True
    return ret

def dumps_json(obj, *args, **kwargs):
    """Dump `obj` to a JSON string using `json_tricks`.

    Temporarily sets `SHOW_SCALAR_WARNING` to false to avoid messages when
    serializing numpy scalars.
    """
    json_tricks.NumpyEncoder.SHOW_SCALAR_WARNING=False
    ret = json_tricks.dumps(obj, *args, **kwargs)
    json_tricks.NumpyEncoder.SHOW_SCALAR_WARNING=True
    return ret

def load_json(filepath_or_buffer, *args, **kwargs):
    """Load an object from a JSON file."""
    if isinstance(filepath_or_buffer, str):
        filepath_or_buffer = open(filepath_or_buffer, 'r')
    return json_tricks.load(filepath_or_buffer, *args, **kwargs)

def loads_json(s, *args, **kwargs):
    """Load an object from a JSON string."""
    return json_tricks.loads(s, *args, **kwargs)

def to_csv(obj, output=None, metadata=dict()):
    """Convert an object to a CSV file.

    TODO: Follow `pandas` insofar as possible except allowing for serializing
    numpy arrays and preserving metadata in comments in the first few lines.
    """
    pass

def seq_to_dict(inner_seq, keys):
    """Convert a sequence of sequences to a sequence of dicts with each dict
    associating each element of the inner sequence to a key."""
    return [{k: elem for k, elem in zip(keys, inner_seq)} for inner_seq in seq]

def string_to_array(s, sep=' ', shape=None, **kwargs):
    """Convert a string to a numpy array.
    For example, `"[1 2 3]"` --> `np.array([1, 2, 3])`

    Useful for working with pandas and CSV arrays.
    """
    # Replace brackets because numpy doesn't like them
    raw = s.replace('[', '').replace(']', '')
    arr = np.fromstring(raw, sep=sep)
    # Optionally reshape
    if shape is not None:
        return arr.reshape(shape)
    else:
        return arr
