"""Utilities for working with data, e.g., CSV files, JSON files, and so on."""
import io
import os
import zlib
import pandas as pd
import json_tricks as jt
import msgpack


def dump_msgpack(obj, path_or_buf=None, **kwargs):
    """Dump an object to msgpack, using zlib for compression."""
    # Dump the object using its built-in method or msgpack's default
    if hasattr(obj, 'to_msgpack') and callable(obj.to_msgpack):
        msg = obj.to_msgpack(**kwargs)
    else:
        msg = msgpack.dumps(obj)
    elem = zlib.compress(msg)

    if path_or_buf is None:
        return elem
    elif isinstance(path_or_buf, str):
        with open(path_or_buf, 'wb') as fh:
            fh.write(elem)
    else:
        path_or_buf.write(elem)

def load_msgpack(path_or_buf, use_pandas=True):
    """Load a compressed msgpack object, by default using Pandas"""
    def loader(path_or_buf):
        """Load either from a buffer or a file path."""
        if isinstance(path_or_buf, str):
            try:
                exists = os.path.exists(path_or_buf)
            except (TypeError, ValueError):
                exists = False

            # If it's a filepath, open and read it, else treat as bytes
            if exists:
                return open(path_or_buf, 'rb').read()
            else:
                return bytes(path_or_buf, 'ascii')

        # If it's a bytes object, just return it
        if isinstance(path_or_buf, bytes):
            return path_or_buf

        # Buffer-like
        if hasattr(path_or_buf, 'read') and callable(path_or_buf.read):
            return path_or_buf.read()
        raise ValueError("Could not load `path_or_buf`")

    elem = zlib.decompress(loader(path_or_buf))
    if use_pandas:
        return pd.read_msgpack(elem)
    else:
        return msgpack.loads(elem)

def dump_json(obj, path_or_buf=None, **kwargs):
    """Serialize and compress an object using JSON."""
    if hasattr(obj, 'to_json') and callable(obj.to_json):
        msg = obj.to_json(**kwargs)
    else:
        msg = jt.dumps(obj)
    elem = zlib.compress(bytes(msg, 'ascii'))

    if path_or_buf is None:
        return elem
    elif isinstance(path_or_buf, str):
        with open(path_or_buf, 'wb') as fh:
            fh.write(elem)
    else:
        path_or_buf.write(elem)

def load_json(path_or_buf, use_pandas=True):
    """Load a compressed JSON object, by default using Pandas"""
    def loader(path_or_buf):
        """Load either from a buffer or a file path."""
        if isinstance(path_or_buf, str):
            try:
                exists = os.path.exists(path_or_buf)
            except (TypeError, ValueError):
                exists = False

            # If it's a filepath, open and read it, else treat as bytes
            if exists:
                return open(path_or_buf, 'rb').read()
            else:
                return bytes(path_or_buf, 'ascii')

        # If it's a bytes object, just return it
        if isinstance(path_or_buf, bytes):
            return path_or_buf

        # Buffer-like
        if hasattr(path_or_buf, 'read') and callable(path_or_buf.read):
            return path_or_buf.read()
        raise ValueError("Could not load `path_or_buf`")

    elem = zlib.decompress(loader(path_or_buf))
    if use_pandas:
        return pd.read_json(elem)
    else:
        return jt.loads(str(elem, 'ascii'))


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
